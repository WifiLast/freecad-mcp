"""GPU linear-elastic voxel solver (torch, ROCm or CUDA), run as a helper process by topopt.GPUBackend.

Matrix-free: K u = scatter_e( E_e * KE @ u[edof_e] ) - no global matrix is assembled, the hex grid
is regular so one 24x24 KE serves all elements. CG batched over load cases, warm-started from the
previous solve (SIMP changes densities only a little per iteration). Preconditioner: geometric
multigrid V-cycle when the voxel indices are given (iterations stay ~flat with resolution and
slenderness), else Jacobi.

Private helper: protocol on stdin/stdout. Shared helper: `--socket PATH [--idle S]`, same protocol per connection.
Protocol: 8-byte little-endian length + pickled dict, one reply per request.
  {"cmd":"setup","sid":..,"edof":(m,24) int,"free":(ndof,) bool,"F":(ndof,nlc),"KE":(24,24),
   "eijk":(m,3) int voxel indices (optional, enables multigrid),"corners":(8,3) local node offsets}
  {"cmd":"solve","sid":..,"Ee":(m,),"tol":1e-6,"energy":True,"return_U":False}
  {"cmd":"mesh_optimize","vertices":(V,3),"faces":(F,3),"target_faces":..} (CuMesh, see mesh_gpu.py)
  {"cmd":"get_U","sid":..} | {"cmd":"drop","sid":..} | {"cmd":"info"} | {"cmd":"quit"}
"""
import os
import pickle
import struct
import sys
import threading
import time

import numpy as np
import torch

DEV = torch.device(os.environ.get("FREECAD_GPU_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
DT = torch.float64


class Level:
    """One coarse multigrid level: voxels of 2h with full 24x24 Galerkin element matrices
    Kc_E = sum_children Q_o^T K_child Q_o (exact P^T K P for element-wise trilinear prolongation)."""

    def __init__(self, eijk_f, enodes_f, corners):
        cor = np.asarray(corners)
        par_ijk = eijk_f // 2
        oct_ = eijk_f % 2
        self.eijk, par = np.unique(par_ijk, axis=0, return_inverse=True)
        par = par.ravel()
        N = self.eijk.max(0) + 2
        keys = np.empty((len(self.eijk), 8), np.int64)
        for k, (a, b, c) in enumerate(cor):
            keys[:, k] = ((self.eijk[:, 0] + a) * N[1] + self.eijk[:, 1] + b) * N[2] + self.eijk[:, 2] + c
        _, inv = np.unique(keys, return_inverse=True)
        self.enodes = inv.reshape(keys.shape)
        self.nnode = int(self.enodes.max()) + 1
        self.m = len(self.eijk)
        # W[o][k, K]: weight of coarse corner K at fine corner k of a child in octant o
        octs = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)])
        W = np.zeros((8, 8, 8))
        for o, ov in enumerate(octs):
            t = (ov[None, :] + cor) / 2.0  # fine corner positions in coarse-element units
            W[o] = np.prod(np.where(cor[None, :, :] == 1, t[:, None, :], 1 - t[:, None, :]), axis=2)
        oid = oct_[:, 0] * 4 + oct_[:, 1] * 2 + oct_[:, 2]
        # global prolongation triples (fine node, coarse node, weight), deduplicated
        k, K = np.nonzero(np.ones((8, 8)))
        fn = enodes_f[:, k].ravel()
        cn = self.enodes[par][:, K].ravel()
        w = W[oid][:, k, K].ravel()
        nz = w > 1e-12
        fn, cn, w = fn[nz], cn[nz], w[nz]
        _, first = np.unique(fn.astype(np.int64) * self.nnode + cn, return_index=True)
        self.pf = torch.as_tensor(fn[first], device=DEV)
        self.pc = torch.as_tensor(cn[first], device=DEV)
        self.pw = torch.as_tensor(w[first], dtype=DT, device=DEV)
        self.Q = torch.as_tensor(np.stack([np.kron(W[o], np.eye(3)) for o in range(8)]), dtype=DT, device=DEV)
        self.groups = [(torch.as_tensor(np.flatnonzero(oid == o), device=DEV),
                        torch.as_tensor(par[oid == o], device=DEV)) for o in range(8)]
        ed = (3 * self.enodes[:, :, None] + np.arange(3)[None, None, :]).reshape(self.m, 24)
        self.edof = torch.as_tensor(ed, dtype=torch.long, device=DEV)
        self.flat = self.edof.reshape(-1)
        self.ndof = 3 * self.nnode
        self.K = None
        self.Minv = None

    def build(self, child_K):
        """child_K(idx) -> (len(idx),24,24) element matrices of the finer level."""
        Kc = torch.zeros((self.m, 24, 24), dtype=DT, device=DEV)
        for o, (idx, par) in enumerate(self.groups):
            Q = self.Q[o]
            for ch in torch.split(torch.arange(len(idx), device=DEV), 200000):
                Kc.index_add_(0, par[ch], Q.T @ child_K(idx[ch]) @ Q)
        self.K = Kc
        d = torch.zeros(self.ndof, dtype=DT, device=DEV)
        d.index_add_(0, self.flat, torch.diagonal(Kc, dim1=1, dim2=2).reshape(-1))
        self.active = d > 1e-14 * d.max()
        self.Minv = torch.where(self.active, 1.0 / torch.where(self.active, d, torch.ones_like(d)),
                                torch.zeros_like(d))[:, None]

    def matvec(self, X):
        Y = torch.zeros_like(X)
        Y.index_add_(0, self.flat, torch.einsum("mij,mjc->mic", self.K, X[self.edof]).reshape(-1, X.shape[1]))
        return Y

    def prolong(self, Xc, nf):
        nlc = Xc.shape[1]
        Xf = torch.zeros((nf, 3, nlc), dtype=DT, device=DEV)
        Xf.index_add_(0, self.pf, self.pw[:, None, None] * Xc.view(-1, 3, nlc)[self.pc])
        return Xf.view(-1, nlc)

    def restrict(self, Xf):
        nlc = Xf.shape[1]
        Xc = torch.zeros((self.nnode, 3, nlc), dtype=DT, device=DEV)
        Xc.index_add_(0, self.pc, self.pw[:, None, None] * Xf.view(-1, 3, nlc)[self.pf])
        return Xc.view(-1, nlc)

    def factor_dense(self):
        A = torch.zeros((self.ndof, self.ndof), dtype=DT, device=DEV)
        i = self.edof[:, :, None].expand(-1, 24, 24).reshape(-1)
        j = self.edof[:, None, :].expand(-1, 24, 24).reshape(-1)
        A.index_put_((i, j), self.K.reshape(-1), accumulate=True)
        idle = ~self.active
        A[idle, :] = 0
        A[:, idle] = 0
        A[idle, idle] = 1.0
        jit = 1e-12 * torch.diagonal(A).abs().mean()
        for _ in range(6):
            L, info = torch.linalg.cholesky_ex(A + jit * torch.eye(self.ndof, dtype=DT, device=DEV))
            if int(info) == 0:
                break
            jit *= 100
        self.L = L

    def coarse_solve(self, B):
        return torch.cholesky_solve(B * self.active[:, None], self.L) * self.active[:, None]


class System:
    def __init__(self, edof, free, F, KE, eijk=None, corners=None):
        self.edof = torch.as_tensor(np.ascontiguousarray(edof), dtype=torch.long, device=DEV)
        self.flat = self.edof.reshape(-1)
        self.mask = torch.as_tensor(free, dtype=DT, device=DEV)[:, None]
        self.F = torch.as_tensor(F, dtype=DT, device=DEV) * self.mask
        self.KE = torch.as_tensor(KE, dtype=DT, device=DEV)
        self.dKE = torch.diagonal(self.KE).clone()
        self.ndof, self.nlc = self.F.shape
        self.m = self.edof.shape[0]
        self.U = torch.zeros_like(self.F)
        self.Ee = None
        self.levels = []
        if eijk is not None and self.m > int(os.environ.get("FREECAD_MG_MIN_ELEMENTS", "3000")):
            enodes = np.asarray(edof)[:, ::3] // 3
            e = np.asarray(eijk)
            while len(self.levels) < 8:
                lv = Level(e, enodes, corners)
                self.levels.append(lv)
                if lv.m <= int(os.environ.get("FREECAD_MG_COARSE", "1500")):
                    break
                e, enodes = lv.eijk, lv.enodes

    def matvec(self, X):
        """K X for X (ndof, nlc), with fixed dofs eliminated (identity-free: rows/cols masked)."""
        X = X * self.mask
        ue = X[self.edof]  # (m,24,nlc)
        fe = torch.einsum("ij,mjc->mic", self.KE, ue) * self.Ee[:, None, None]
        Y = torch.zeros_like(X)
        Y.index_add_(0, self.flat, fe.reshape(-1, X.shape[1]))
        return Y * self.mask

    def diag(self):
        d = torch.zeros(self.ndof, dtype=DT, device=DEV)
        d.index_add_(0, self.flat, (self.Ee[:, None] * self.dKE[None, :]).reshape(-1))
        d = torch.where(self.mask[:, 0] > 0, d, torch.ones_like(d))
        return d[:, None]

    def fine_K(self, idx):
        """Masked fine element matrices E_e * (d d^T o KE) for elements idx."""
        d = self.mask[:, 0][self.edof[idx]]  # (n,24)
        return self.Ep[idx, None, None] * self.KE[None] * (d[:, :, None] * d[:, None, :])

    def setup_mg(self):
        # the preconditioner sees voids no softer than MG_EMIN * max(E): near-free solid islands held by
        # 1e-9 voids would make the coarse levels nearly singular (CG breakdown); CG still solves the true K
        self.Ep = self.Ee.clamp_min(float(os.environ.get("FREECAD_MG_EMIN", "0")) * float(self.Ee.max()))
        prev = self.fine_K
        for lv in self.levels:
            lv.build(prev)
            prev = (lambda L: (lambda idx: L.K[idx]))(lv)
        self.levels[-1].factor_dense()
        # damped Jacobi is only a convergent smoother (-> SPD preconditioner) for omega < 2/lambda_max(D^-1 A);
        # lambda_max is ~4-6 here and shifts with the densities, so estimate it per level and solve
        self.omega = []
        for lvl in range(len(self.levels)):
            A, Minv = self._op(lvl)
            x = torch.randn(Minv.shape[0], 1, dtype=DT, device=DEV) * (Minv > 0)
            lam = 1.0
            for _ in range(12):
                y = Minv * A(x)
                lam = float(torch.linalg.norm(y) / torch.linalg.norm(x).clamp_min(1e-300))
                x = y / torch.linalg.norm(y).clamp_min(1e-300)
            self.omega.append(1.2 / (1.1 * lam))

    def _op(self, lvl):
        if lvl == 0:
            return self.matvec, self.Minv
        L = self.levels[lvl - 1]
        return L.matvec, L.Minv

    def vcycle(self, B, lvl=0, nu=2):
        """Symmetric V-cycle (damped Jacobi pre/post smoothing) as the CG preconditioner."""
        A, Minv = self._op(lvl)
        omega = self.omega[lvl]
        X = omega * Minv * B
        for _ in range(nu - 1):
            X = X + omega * Minv * (B - A(X))
        R = B - A(X)
        lv = self.levels[lvl]
        if lvl + 1 == len(self.levels):
            Ec = lv.coarse_solve(lv.restrict(R))
        else:
            Ec = self.vcycle(lv.restrict(R), lvl + 1, nu)
        X = X + lv.prolong(Ec, B.shape[0] // 3)
        X = X * (self.mask if lvl == 0 else self.levels[lvl - 1].active[:, None])
        for _ in range(nu):
            X = X + omega * Minv * (B - A(X))
        return X

    def solve(self, Ee, tol=1e-6, maxiter=20000):
        self.Ee = torch.as_tensor(Ee, dtype=DT, device=DEV)
        Minv = 1.0 / self.diag()
        self.Minv = Minv * self.mask
        if self.levels:
            self.setup_mg()
            M = self.vcycle
            check = 1
        else:
            M = lambda R: Minv * R
            check = 10
        X = self.U * self.mask
        R = self.F - self.matvec(X)
        Z = M(R)
        P = Z.clone()
        rz = (R * Z).sum(0)
        bnorm = torch.linalg.norm(self.F, dim=0).clamp_min(1e-300)
        it = 0
        for it in range(1, maxiter + 1):
            AP = self.matvec(P)
            alpha = rz / (P * AP).sum(0).clamp_min(1e-300)
            X = X + alpha * P
            R = R - alpha * AP
            if it % check == 0 and bool((torch.linalg.norm(R, dim=0) / bnorm <= tol).all()):
                break
            Z = M(R)
            rz_new = (R * Z).sum(0)
            P = Z + (rz_new / rz.clamp_min(1e-300)) * P
            rz = rz_new
        self.U = X
        res = float((torch.linalg.norm(R, dim=0) / bnorm).max())
        return it, res

    def energy(self):
        ue = self.U[self.edof]  # (m,24,nlc)
        return torch.einsum("mic,ij,mjc->m", ue, self.KE, ue)


def _send(w, obj):
    data = pickle.dumps(obj, protocol=4)
    w.write(struct.pack("<Q", len(data)) + data)
    w.flush()


def _recv(r):
    head = r.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    return pickle.loads(r.read(n))


GPU_LOCK = threading.Lock()
STATE = {"last": time.time(), "clients": 0}


def serve(r, w):
    """Request loop for one client; its systems are freed when it disconnects."""
    systems = {}
    try:
        while True:
            req = _recv(r)
            if req is None or req.get("cmd") == "quit":
                break
            STATE["last"] = time.time()
            try:
                with GPU_LOCK:
                    out = handle(req, systems)
            except Exception as e:  # keep serving; the client decides to fall back to CPU
                out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            _send(w, out)
    finally:
        systems.clear()
        if DEV.type == "cuda":
            with GPU_LOCK:
                torch.cuda.empty_cache()


def handle(req, systems):
    cmd = req["cmd"]
    if cmd == "info":
        name = torch.cuda.get_device_name(DEV) if DEV.type == "cuda" else "cpu"
        free, total = torch.cuda.mem_get_info(DEV) if DEV.type == "cuda" else (0, 0)
        return {"ok": True, "device": name, "torch": str(torch.__version__), "hip": str(torch.version.hip),
                "free_mb": int(free // 2**20), "total_mb": int(total // 2**20)}
    if cmd == "setup":
        systems[req["sid"]] = System(req["edof"], req["free"], req["F"], req["KE"], req.get("eijk"), req.get("corners"))
        return {"ok": True}
    if cmd == "solve":
        s = systems[req["sid"]]
        t = time.time()
        it, res = s.solve(req["Ee"], req.get("tol", 1e-6), req.get("maxiter", 20000))
        out = {"ok": True, "iterations": it, "residual": res, "compliance": float((s.F * s.U).sum())}
        if req.get("energy", True):
            out["ce"] = s.energy().cpu().numpy()
        if req.get("return_U"):
            out["U"] = s.U.cpu().numpy()
        if DEV.type == "cuda":
            torch.cuda.synchronize(DEV)
        out["time_s"] = time.time() - t
        return out
    if cmd == "mesh_info":
        import mesh_gpu
        ok, err = mesh_gpu.available()
        return {"ok": True, "cumesh": ok, "error": err}
    if cmd == "mesh_optimize":
        import mesh_gpu
        v, f, rep = mesh_gpu.optimize(req["vertices"], req["faces"], device=str(DEV),
                                      **{k: req[k] for k in ("target_faces", "repair", "fill_holes", "min_component_area",
                                                             "unify_orientation", "simplify_options") if k in req})
        return {"ok": True, "vertices": v, "faces": f, "report": rep}
    if cmd == "get_U":
        return {"ok": True, "U": systems[req["sid"]].U.cpu().numpy()}
    if cmd == "drop":
        systems.pop(req["sid"], None)
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        return {"ok": True}
    return {"ok": False, "error": f"unknown cmd {cmd}"}


def serve_socket(path, idle):
    """Shared helper: torch + GPU context stay warm between jobs; exits after `idle` s unused."""
    import socket
    if os.path.exists(path):
        os.remove(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o600)
    srv.listen(8)
    srv.settimeout(5.0)

    def client(conn):
        STATE["clients"] += 1
        try:
            serve(conn.makefile("rb"), conn.makefile("wb"))
        finally:
            STATE["clients"] -= 1
            STATE["last"] = time.time()
            conn.close()

    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                if STATE["clients"] == 0 and time.time() - STATE["last"] > idle:
                    break
                continue
            threading.Thread(target=client, args=(conn,), daemon=True).start()
    finally:
        srv.close()
        if os.path.exists(path):
            os.remove(path)


def main():
    if "--socket" in sys.argv:
        path = sys.argv[sys.argv.index("--socket") + 1]
        idle = float(sys.argv[sys.argv.index("--idle") + 1]) if "--idle" in sys.argv else 600.0
        serve_socket(path, idle)
    else:
        serve(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    main()
