"""Voxel (hex8) SIMP topology optimisation + linear-elastic voxel FEM. numpy/scipy/pyamg only.

Grid: element (ix,iy,iz) occupies [o + i*h, o + (i+1)*h]; node (ix,iy,iz) at o + i*h.
Only "active" elements (inside the original part) take part; everything else is empty space.
Units: mm, N, MPa.
"""
import os
import subprocess
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    import sksparse.cholmod as _cholmod
except Exception:  # pragma: no cover
    _cholmod = None

DIRECT_MAX_DOF = 450000  # above this the direct factorisation needs too much RAM -> CG

# local corner order of a hex8 element
CORNERS = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], int)


def elasticity_D(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def _B(xi, eta, zeta, h=1.0):
    s = 2 * CORNERS - 1
    dN = np.zeros((8, 3))
    dN[:, 0] = s[:, 0] * (1 + eta * s[:, 1]) * (1 + zeta * s[:, 2]) / 8
    dN[:, 1] = s[:, 1] * (1 + xi * s[:, 0]) * (1 + zeta * s[:, 2]) / 8
    dN[:, 2] = s[:, 2] * (1 + xi * s[:, 0]) * (1 + eta * s[:, 1]) / 8
    dN *= 2.0 / h  # d/dx = d/dxi * 2/h
    B = np.zeros((6, 24))
    for i in range(8):
        dx, dy, dz = dN[i]
        B[:, 3 * i:3 * i + 3] = [[dx, 0, 0], [0, dy, 0], [0, 0, dz], [dy, dx, 0], [0, dz, dy], [dz, 0, dx]]
    return B


def hex_KE(D, h=1.0):
    """Stiffness of a cube of side h for the 6x6 material matrix D (a float = isotropic nu, E=1); 2x2x2 Gauss."""
    if np.ndim(D) == 0:
        D = elasticity_D(1.0, float(D))
    g = 1 / np.sqrt(3)
    KE = np.zeros((24, 24))
    detJ = (h / 2) ** 3
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = _B(xi, eta, zeta, h)
                KE += B.T @ D @ B * detJ
    return KE


def von_mises(s):
    sx, sy, sz, txy, tyz, tzx = s.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))


# Voigt shear components lying on the layer plane of build axis a: xy=3, yz=4, zx=5
_LAYER_SHEAR = {0: (3, 5), 1: (3, 4), 2: (4, 5)}


def layer_stress(s, axis, yield_z, shear_z):
    """Interlayer stresses of (n,6) Voigt stresses: sigma_n = tension across the layers (compression
    does not open them), tau = shear on the layer plane; utilisation = quadratic interaction
    (sigma_n/yield_z)^2 + (tau/shear_z)^2, reported as its root (1 = layer bond fails)."""
    sn = np.maximum(s[:, axis], 0.0)
    i, j = _LAYER_SHEAR[axis]
    tau = np.hypot(s[:, i], s[:, j])
    return sn, tau, np.sqrt((sn / yield_z) ** 2 + (tau / shear_z) ** 2)


class Grid:
    def __init__(self, origin, h, shape):
        self.o = np.asarray(origin, float)
        self.h = float(h)
        self.n = tuple(int(v) for v in shape)  # elements per axis
        nx, ny, nz = self.n
        self.nn = (nx + 1, ny + 1, nz + 1)

    def node_id(self, ix, iy, iz):
        return (ix * self.nn[1] + iy) * self.nn[2] + iz

    def node_xyz(self, ids):
        ids = np.asarray(ids)
        iz = ids % self.nn[2]
        iy = (ids // self.nn[2]) % self.nn[1]
        ix = ids // (self.nn[1] * self.nn[2])
        return self.o + self.h * np.stack([ix, iy, iz], -1)

    def elem_nodes(self, eidx):
        """eidx: (m,3) element index triples -> (m,8) global node ids."""
        e = np.asarray(eidx)
        out = np.empty((len(e), 8), np.int64)
        for k, (a, b, c) in enumerate(CORNERS):
            out[:, k] = self.node_id(e[:, 0] + a, e[:, 1] + b, e[:, 2] + c)
        return out

    def elem_centers(self, eidx):
        return self.o + self.h * (np.asarray(eidx) + 0.5)


class VoxelFEM:
    """Linear elastic FE system over a set of active voxels with node BCs."""

    def __init__(self, grid, elems, D, fixed, loadcases, backend=None):
        """elems: (m,3) element indices. D: unit material matrix (or isotropic nu); element stiffness
        scales with Ee. fixed: list of (global_node_ids, 'xyz').
        loadcases: list of dict global_node_id -> force vec (3,)."""
        self.g = grid
        self.elems = np.asarray(elems)
        self.m = len(self.elems)
        gnodes = grid.elem_nodes(self.elems)
        uniq, inv = np.unique(gnodes, return_inverse=True)
        self.gnode = uniq  # local node -> global node id
        self.enodes = inv.reshape(gnodes.shape)  # (m,8) local node ids
        self.nnode = len(uniq)
        self.ndof = 3 * self.nnode
        self.edof = (3 * self.enodes[:, :, None] + np.arange(3)[None, None, :]).reshape(self.m, 24)
        self.KE = hex_KE(D, grid.h)
        self.iK = np.repeat(self.edof, 24, axis=1).ravel()
        self.jK = np.tile(self.edof, (1, 24)).ravel()
        g2l = {int(gn): i for i, gn in enumerate(uniq)}
        fixed_mask = np.zeros(self.ndof, bool)
        for ids, dofs in fixed:
            for gn in ids:
                li = g2l.get(int(gn))
                if li is None:
                    continue
                for a, ax in enumerate("xyz"):
                    if ax in dofs:
                        fixed_mask[3 * li + a] = True
        if not fixed_mask.any():
            raise ValueError("no fixed node lies on the active voxels")
        self.free = np.flatnonzero(~fixed_mask)
        F = np.zeros((self.ndof, len(loadcases)))
        lost = 0.0
        for c, lc in enumerate(loadcases):
            for gn, f in lc.items():
                li = g2l.get(int(gn))
                if li is None:
                    lost += float(np.linalg.norm(f))
                    continue
                F[3 * li:3 * li + 3, c] += f
        self.F = F
        self.lost_force = lost
        self._U = np.zeros_like(F)
        self._ce = None
        self._comp = None
        self.solver_stats = {"backend": "cpu", "solves": 0, "cg_iterations": 0, "time_s": 0.0}
        self.backend = backend
        self.sid = None
        if backend is not None:
            free_mask = np.zeros(self.ndof, bool)
            free_mask[self.free] = True
            self.sid = backend.setup(self.edof, free_mask, F, self.KE, eijk=self.elems)
            self.solver_stats["backend"] = backend.device_name

    @property
    def U(self):
        if self._U is None:  # solved on the GPU, fetch lazily
            self._U = self.backend.fetch_U(self.sid)
        return self._U

    def close(self):
        if self.backend is not None and self.sid is not None:
            self.backend.drop(self.sid)
            self.sid = None

    def solve(self, Ee, tol=None):
        """tol only steers the iterative (GPU/CG) paths; direct factorisations are exact."""
        t0 = time.time()
        self.solver_stats["solves"] += 1
        if self.backend is not None:
            r = self.backend.solve(self.sid, Ee, tol=tol or self.backend.tol)
            self._ce, self._comp, self._U = r["ce"], r["compliance"], None
            self.solver_stats["cg_iterations"] += r["iterations"]
            self.solver_stats["time_s"] += time.time() - t0
            return
        self._ce = self._comp = None
        sK = (self.KE.ravel()[None, :] * Ee[:, None]).ravel()
        K = sp.coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsr()
        K = K[self.free][:, self.free]
        K = (K + K.T) * 0.5
        nf = len(self.free)
        b = self.F[self.free]
        if nf <= DIRECT_MAX_DOF:
            if _cholmod is not None:  # supernodal CHOLMOD: ~10x faster than SuperLU on 3D grids
                try:
                    self._U[self.free] = _cholmod.cho_factor(K.tocsc(), supernodal_mode="supernodal").solve(b).reshape(b.shape)
                    self.solver_stats["time_s"] += time.time() - t0
                    return
                except Exception:
                    pass
            lu = spla.splu(K.tocsc(), permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
            for c in range(b.shape[1]):
                self._U[self.free, c] = lu.solve(b[:, c])
            self.solver_stats["time_s"] += time.time() - t0
            return
        # very large grids: Jacobi-preconditioned CG, warm-started from the previous solution
        d = 1.0 / K.diagonal()
        M = spla.LinearOperator(K.shape, lambda v: d * v)
        for c in range(b.shape[1]):
            x, info = spla.cg(K, b[:, c], x0=self._U[self.free, c], rtol=tol or 1e-7, maxiter=20000, M=M)
            self._U[self.free, c] = x
        self.solver_stats["time_s"] += time.time() - t0

    def elem_energy(self):
        """sum over load cases of u_e^T KE u_e (E=1) -> (m,)"""
        if self._ce is not None:
            return self._ce
        ce = np.zeros(self.m)
        for c in range(self.F.shape[1]):
            ue = self.U[self.edof, c]
            ce += np.einsum("ij,jk,ik->i", ue, self.KE, ue)
        return ce

    def compliance(self):
        if self._comp is not None:
            return self._comp
        return float(np.sum(self.F * self.U))

    def stresses(self, D_real, layer=None):
        """Element-center von Mises (max over load cases) for the full material matrix D_real.
        layer=(axis, yield_z, shear_z) also returns the interlayer sigma_n, tau and utilisation."""
        B0 = _B(0, 0, 0, self.g.h)
        S = D_real @ B0  # 6x24
        vm = np.zeros(self.m)
        lay = {"sigma_n": np.zeros(self.m), "tau": np.zeros(self.m), "util": np.zeros(self.m)} if layer else None
        for c in range(self.F.shape[1]):
            s = self.U[self.edof, c] @ S.T
            vm = np.maximum(vm, von_mises(s))
            if layer:
                for k, v in zip(("sigma_n", "tau", "util"), layer_stress(s, *layer)):
                    lay[k] = np.maximum(lay[k], v)
        return vm, lay

    def max_disp(self):
        d = np.linalg.norm(self.U.reshape(self.nnode, 3, -1), axis=1)  # (nnode, nlc)
        i = np.unravel_index(np.argmax(d), d.shape)
        return float(d[i]), self.g.node_xyz(self.gnode[i[0]])


def density_filter(grid, elems, rmin_elems):
    c = np.asarray(elems, float)
    tree = cKDTree(c)
    D = tree.sparse_distance_matrix(tree, rmin_elems, output_type="coo_matrix")
    w = np.maximum(rmin_elems - D.data, 0)
    H = sp.coo_matrix((w, (D.row, D.col)), shape=(len(c), len(c))).tocsr()
    H = H + sp.identity(len(c), format="csr") * rmin_elems  # diagonal (distance 0 not stored)
    Hs = np.asarray(H.sum(1)).ravel()
    return H, Hs


def mirror_map(grid, elems, axes):
    """For symmetry planes through the grid centre: index of mirrored element (or -1)."""
    lookup = {tuple(e): i for i, e in enumerate(map(tuple, elems))}
    maps = []
    for ax in axes:
        a = "xyz".index(ax)
        m = np.full(len(elems), -1, np.int64)
        for i, e in enumerate(elems):
            f = list(e)
            f[a] = grid.n[a] - 1 - f[a]
            m[i] = lookup.get(tuple(f), -1)
        maps.append(m)
    return maps


def _symmetrize(v, maps):
    for m in maps:
        ok = m >= 0
        w = v.copy()
        w[ok] = 0.5 * (v[ok] + v[m[ok]])
        v = w
    return v


class AMFilter:
    """Langelaar's additive-manufacturing filter (3D, 45 deg): printed density of a voxel is
    smin(own density, smax(printed density of its 5 supports in the layer below: directly below +
    the 4 face neighbours of that one)). Applied bottom-up after the density filter, so voxels can
    only be solid where the layer below carries them - overhangs above 45 deg vanish from the design.
    Supports: the build plate (first layer) and part-boundary voxels (directly below outside the part =
    the original geometry's own down-facing surface) count as solid; passive voxels stay 1."""
    P, XI0, EPS = 40.0, 0.5, 1e-4

    def __init__(self, grid, elems, passive, axis=2, sign=1):
        e = np.asarray(elems)
        n = grid.n
        lookup = np.full(n, -1, np.int64)
        lookup[e[:, 0], e[:, 1], e[:, 2]] = np.arange(len(e))
        layer = e[:, axis] if sign > 0 else n[axis] - 1 - e[:, axis]
        self.layer = layer
        self.lookup, self.n, self.axis, self.sign = lookup, n, axis, sign
        sup = np.full((len(e), 5), -1, np.int64)  # >=0 element, -1 none, -2 solid (bed / part boundary)
        o1, o2 = [a for a in range(3) if a != axis]
        offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        for k, (d1, d2) in enumerate(offs):
            q = e.copy()
            q[:, axis] -= sign
            q[:, o1] += d1
            q[:, o2] += d2
            inside = np.all((q >= 0) & (q < np.array(n)), axis=1)
            idx = np.full(len(e), -1, np.int64)
            idx[inside] = lookup[q[inside, 0], q[inside, 1], q[inside, 2]]
            if k == 0:  # directly below: bed or outside the part -> carried
                idx[idx < 0] = -2
            sup[:, k] = idx
        sup[layer == 0, 0] = -2
        self.sup = sup
        self.Q = self.P + np.log(np.maximum((sup != -1).sum(1), 1)) / np.log(self.XI0)
        self.passive = np.asarray(passive, bool)
        order = np.argsort(layer, kind="stable")
        des = order[~self.passive[order]]
        bounds = np.searchsorted(layer[des], np.arange(layer.max() + 2))
        self.layers = [des[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]

    def _vals(self, xi, ids):
        sp_ = self.sup[ids]
        return np.where(sp_ >= 0, xi[np.maximum(sp_, 0)], np.where(sp_ == -2, 1.0, 0.0))

    def forward(self, xt):
        xi = np.array(xt, float)
        xi[self.passive] = 1.0
        self._cache = []
        for ids in self.layers:
            v = self._vals(xi, ids)
            S = np.maximum((v ** self.P).sum(1), 1e-300)
            Xi = S ** (1.0 / self.Q[ids])
            d = xt[ids] - Xi
            r = np.sqrt(d * d + self.EPS)
            xi[ids] = 0.5 * (xt[ids] + Xi - r + np.sqrt(self.EPS))
            self._cache.append((v, S, d / r))
        return xi

    def backward(self, g):
        """d f / d xt from d f / d xi (call after forward with the same xt). g: (m,) or (m,c)."""
        g = np.asarray(g, float)
        one = g.ndim == 1
        lam = (g[:, None] if one else g).copy()
        out = lam.copy()
        for ids, (v, S, dr) in zip(reversed(self.layers), reversed(self._cache)):
            l_ = lam[ids]
            out[ids] = l_ * (0.5 * (1 - dr))[:, None]
            dXi = l_ * (0.5 * (1 + dr))[:, None]
            Q = self.Q[ids]
            dv = (self.P / Q)[:, None] * v ** (self.P - 1) * (S ** (1.0 / Q - 1))[:, None]  # (k,5)
            sp_ = self.sup[ids]
            ok = sp_ >= 0
            np.add.at(lam, sp_[ok], (dXi[:, None, :] * dv[:, :, None])[ok])
        return out[:, 0] if one else out


def unsupported(elems, solid, am):
    """Solid voxels whose 5 supports in the layer below are all empty (and not bed/part boundary)."""
    v = np.where(am.sup >= 0, solid[np.maximum(am.sup, 0)], am.sup == -2)
    return solid & ~v.any(1)


def self_support_fill(elems, solid, am):
    """Make every solid voxel printable without support material by adding material only. Top-down,
    a voxel with nothing under it gets the one of its 5 supports closest to existing material in the
    layer below, so under a flat ceiling a 45 deg roof grows inward from the walls (tent / teardrop
    void) instead of a horizontal bridge. Returns (solid, n_added)."""
    e = np.asarray(elems)
    A = np.zeros(am.n, bool)
    A[e[solid, 0], e[solid, 1], e[solid, 2]] = True
    act = am.lookup >= 0
    A, act = np.moveaxis(A, am.axis, 0), np.moveaxis(act, am.axis, 0)
    if am.sign < 0:
        A, act = A[::-1], act[::-1]
    A = A.copy()
    n0 = int(A.sum())
    offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    for L in range(A.shape[0] - 1, 0, -1):
        cur, below, actb = A[L], A[L - 1], act[L - 1]
        sup = below | ~actb  # directly below: solid, or outside the part (its own down-facing surface)
        for d1, d2 in offs[1:]:
            sup = sup | _shift2(below, d1, d2)
        need = np.argwhere(cur & ~sup)
        if not len(need):
            continue
        dist = ndimage.distance_transform_edt(~below) if below.any() else np.zeros(below.shape)
        cand = need[:, None, :] + np.array(offs)[None]  # (k,5,2), directly below first
        ok = np.all((cand >= 0) & (cand < np.array(below.shape)), axis=2)
        c0, c1 = np.clip(cand[..., 0], 0, below.shape[0] - 1), np.clip(cand[..., 1], 0, below.shape[1] - 1)
        ok &= actb[c0, c1]
        score = np.where(ok, dist[c0, c1], np.inf)
        k = np.argmin(score, axis=1)  # ties -> directly below
        r = np.arange(len(need))
        below[c0[r, k], c1[r, k]] = True
    if am.sign < 0:
        A = A[::-1]
    A = np.moveaxis(A, 0, am.axis)
    out = A[e[:, 0], e[:, 1], e[:, 2]]
    return out, int(out.sum()) - int(np.asarray(solid).sum())


def _shift2(a, d1, d2):
    """b[i, j] = a[i + d1, j + d2] (False outside)."""
    b = np.zeros_like(a)
    n1, n2 = a.shape
    b[max(0, -d1):n1 - max(0, d1), max(0, -d2):n2 - max(0, d2)] = a[max(0, d1):n1 - max(0, -d1), max(0, d2):n2 - max(0, -d2)]
    return b


def simp(fem, passive, volfrac, penal=3.0, rmin=1.5, max_iter=60, tol=0.01, x0=None, sym_maps=None,
         emin=1e-6, move=0.2, log=None, solve_tol=1e-4, am=None):
    """Minimise compliance s.t. total solid fraction == volfrac (passive elements count as solid).
    am: AMFilter -> overhang-free (self-supporting) design."""
    m = fem.m
    passive = np.asarray(passive, bool)
    design = ~passive
    n_target = volfrac * m - passive.sum()
    if n_target <= 0:
        raise ValueError(f"volume_fraction {volfrac} is smaller than the kept (passive) region "
                         f"{passive.sum() / m:.3f}; raise it or reduce keep_thickness")
    vf_design = min(n_target / max(design.sum(), 1), 1.0)
    H, Hs = density_filter(fem.g, fem.elems, rmin)
    x = np.full(m, vf_design) if x0 is None else np.clip(x0.copy(), 1e-3, 1)
    x[passive] = 1.0
    hist = []
    t0 = time.time()
    xPhys = x

    def phys(xd):
        xt = (H @ xd) / Hs
        xt[passive] = 1.0
        return am.forward(xt) if am is not None else xt

    for it in range(1, max_iter + 1):
        p = min(penal, 1.5 + (penal - 1.5) * it / 15.0) if x0 is None else penal
        xPhys = phys(x)
        Ee = emin + xPhys ** p * (1 - emin)
        fem.solve(Ee, tol=solve_tol)  # OC needs sensitivities, not 1e-7 accurate displacements
        ce = fem.elem_energy()
        c = float(np.sum(Ee * ce))
        dc = -p * xPhys ** (p - 1) * (1 - emin) * ce
        dv = np.ones(m)
        if am is not None:
            dc, dv = am.backward(np.stack([dc, dv], 1)).T
            dv = np.maximum(dv, 1e-9)
        dc = H @ (dc / Hs)
        dv = H @ (dv / Hs)
        if sym_maps:
            dc = _symmetrize(dc, sym_maps)
        # optimality criteria with bisection on the Lagrange multiplier
        l1, l2 = 0.0, 1e9
        dcn = np.minimum(dc, -1e-30)
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            xn = np.clip(x * np.sqrt(-dcn / dv / lmid), np.maximum(0.0, x - move), np.minimum(1.0, x + move))
            xn[passive] = 1.0
            xp = phys(xn)
            if xp.sum() > volfrac * m:
                l1 = lmid
            else:
                l2 = lmid
        if sym_maps:
            xn = _symmetrize(xn, sym_maps)
            xn[passive] = 1.0
        change = float(np.max(np.abs(xn - x)))
        x = xn
        hist.append(round(c, 6))
        if log:
            log(f"it {it} c={c:.5g} vol={xPhys.mean():.3f} change={change:.3f}")
        if it >= 10 and change < tol:
            break
    xPhys = phys(x)
    return {"x": x, "xPhys": xPhys, "iterations": it, "compliance_history": hist, "time_s": round(time.time() - t0, 2)}


def threshold_to_volume(xPhys, passive, volfrac):
    """Binary design whose solid count equals volfrac*m (passive always solid)."""
    m = len(xPhys)
    k = int(round(volfrac * m))
    order = np.argsort(-np.where(passive, 2.0, xPhys))
    solid = np.zeros(m, bool)
    solid[order[:max(k, int(passive.sum()))]] = True
    return solid


def keep_connected(grid, elems, solid, anchor):
    """Drop solid islands that touch no anchor element (anchor = BC/passive elements)."""
    arr = np.zeros(grid.n, bool)
    e = elems[solid]
    arr[e[:, 0], e[:, 1], e[:, 2]] = True
    lab, n = ndimage.label(arr)
    if n <= 1:
        return solid
    a = elems[anchor & solid]
    keep_labels = set(np.unique(lab[a[:, 0], a[:, 1], a[:, 2]])) - {0}
    if not keep_labels:
        sizes = ndimage.sum(arr, lab, range(1, n + 1))
        keep_labels = {int(np.argmax(sizes)) + 1}
    ok = np.isin(lab[elems[:, 0], elems[:, 1], elems[:, 2]], list(keep_labels))
    return solid & ok


def evaluate(grid, elems, solid, D, E, fixed, loadcases, backend=None, layer=None):
    """Voxel FEM of a binary design: compliance, max displacement, element von Mises (+ interlayer
    stresses with layer=(axis, yield_z, shear_z)). D: unit material matrix (in-layer E = 1)."""
    sub = elems[solid]
    fem = VoxelFEM(grid, sub, D, fixed, loadcases, backend=backend)
    fem.solve(np.ones(len(sub)) * E)
    vm, lay = fem.stresses(E * np.asarray(D) if np.ndim(D) else elasticity_D(E, D), layer)
    dmax, dat = fem.max_disp()
    out = {"vm": vm, "layer": lay, "disp_max": dmax, "disp_at": dat, "compliance": fem.compliance(),
           "lost_force": fem.lost_force, "solver": dict(fem.solver_stats)}
    fem.close()
    return out


class GPUBackend:
    """Client for gpu_solver.py running in a torch env (ROCm/CUDA).

    Connects to the shared helper on FREECAD_GPU_SOCKET (started by the MCP server, keeps torch and
    the GPU context warm) and otherwise spawns a private helper process for this job."""

    def __init__(self, python=None, tol=1e-6, log=None):
        import socket
        import subprocess
        self.tol = tol
        self._sid = 0
        self.p = self.sock = None
        path = os.environ.get("FREECAD_GPU_SOCKET") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    "work", "gpu.sock")
        if os.path.exists(path):
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(path)
                self._r = self.sock.makefile("rb")
                self._w = self.sock.makefile("wb")
            except OSError:
                self.sock = None
        if self.sock is None:
            here = os.path.dirname(os.path.abspath(__file__))
            python = python or gpu_python()
            if not os.path.exists(python):
                raise RuntimeError(f"GPU python not found: {python}")
            self.p = subprocess.Popen([python, os.path.join(here, "gpu_solver.py")], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=gpu_env(), cwd=here)
            self._r, self._w = self.p.stdout, self.p.stdin
        info = self._call({"cmd": "info"})
        if "cpu" == info.get("device"):
            self.close()
            raise RuntimeError("torch sees no GPU")
        self.info = info
        self.info["shared"] = self.sock is not None
        self.device_name = f"gpu:{info['device']}"

    def _call(self, req):
        import pickle
        import struct
        data = pickle.dumps(req, protocol=4)
        self._w.write(struct.pack("<Q", len(data)) + data)
        self._w.flush()
        head = self._r.read(8)
        if len(head) < 8:
            raise RuntimeError("GPU helper died")
        (n,) = struct.unpack("<Q", head)
        r = pickle.loads(self._r.read(n))
        if not r.get("ok"):
            raise RuntimeError(f"GPU helper: {r.get('error')}")
        return r

    def setup(self, edof, free, F, KE, eijk=None):
        self._sid += 1
        req = {"cmd": "setup", "sid": self._sid, "edof": edof.astype(np.int64), "free": free, "F": F, "KE": KE}
        if eijk is not None and os.environ.get("FREECAD_GPU_MG", "1") != "0":
            req.update(eijk=np.asarray(eijk, np.int64), corners=CORNERS)
        self._call(req)
        return self._sid

    def solve(self, sid, Ee, tol=1e-6):
        return self._call({"cmd": "solve", "sid": sid, "Ee": np.asarray(Ee, float), "tol": tol})

    def fetch_U(self, sid):
        return self._call({"cmd": "get_U", "sid": sid})["U"]

    def mesh_optimize(self, vertices, faces, **opts):
        """CuMesh repair + decimation on the GPU -> (vertices, faces, report)."""
        r = self._call(dict(opts, cmd="mesh_optimize", vertices=np.asarray(vertices, np.float32),
                            faces=np.asarray(faces, np.int32)))
        return r["vertices"], r["faces"], r["report"]

    def drop(self, sid):
        try:
            self._call({"cmd": "drop", "sid": sid})
        except Exception:
            pass

    def close(self):
        try:
            if self.sock is not None:  # shared helper: just hang up (it frees our systems)
                self._w.close()
                self._r.close()
                self.sock.close()
            else:
                self.p.stdin.close()  # EOF ends the private helper
                self.p.wait(timeout=10)
        except Exception:
            if self.p is not None:
                self.p.kill()


GPU_PYTHONS = {"amd": "/mnt/data/miniconda3/envs/py310_amd/bin/python",      # torch ROCm (RX 7900 XTX)
               "nvidia": "/mnt/data/miniconda3/envs/py310_nv/bin/python"}   # torch CUDA (CMP 50HX)


def gpu_backend():
    """FREECAD_GPU_BACKEND: nvidia (default) | amd; an explicit FREECAD_GPU_PYTHON of the AMD env means amd."""
    py = os.environ.get("FREECAD_GPU_PYTHON", "")
    if py and (py == GPU_PYTHONS["amd"] or "amd" in py or "rocm" in py):
        return "amd"
    b = os.environ.get("FREECAD_GPU_BACKEND", "nvidia").lower()
    return "amd" if b in ("amd", "rocm", "hip") else "nvidia"


def gpu_python():
    return os.environ.get("FREECAD_GPU_PYTHON") or GPU_PYTHONS[gpu_backend()]


def _freest_cuda_device():
    """Index (PCI order, as nvidia-smi) of the NVIDIA GPU with the most free memory, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        rows = [tuple(int(v) for v in ln.split(",")) for ln in out.strip().splitlines() if ln.strip()]
        return max(rows, key=lambda r: r[1])[0] if rows else None
    except Exception:
        return None


def gpu_env():
    """Environment for the torch helper: FreeCAD env paths must not leak into the torch env; on the NVIDIA
    backend the helper sees one card (FREECAD_CUDA_DEVICE, else the one with the most free memory)."""
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    for k in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
        env.pop(k, None)
    if gpu_backend() == "nvidia":  # the AMD toolchain environment (shell) must not steer the CUDA torch
        for k in ("TRITON_DEFAULT_BACKEND", "HIP_PLATFORM", "HSA_XNACK", "PYTORCH_ROCM_ARCH", "CC", "CXX"):
            env.pop(k, None)
    if gpu_backend() == "nvidia" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        dev = os.environ.get("FREECAD_CUDA_DEVICE")
        dev = dev if dev is not None else _freest_cuda_device()
        if dev is not None:
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(dev)
    return env
