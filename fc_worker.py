"""Job runner executed in a separate process by freecad_mcp.py:  python fc_worker.py job.json result.json

Each job is {"action": ..., **args}. Keeps FreeCAD crashes / memory away from the MCP server.
"""
import os
import sys

os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_WARNINGS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# FreeCAD's init clears names already defined in __main__, so it must be imported before anything else
import fc_core  # noqa: E402
from fc_core import App, Part, V  # noqa: E402

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- model actions
def act_create_model(a):
    params = a.get("params") or {}
    if a.get("file") and os.path.splitext(a["file"])[1].lower() in (".stl", ".obj", ".ply", ".off"):
        v, f = load_mesh_file(a["file"])
        cap = int(a.get("mesh_max_faces", 20000))
        src = {"file": a["file"], "mesh_faces_in": len(f)}
        v, f, rep = mesh_optimize(v, f, target_faces=cap if len(f) > cap else None, device=a.get("device"))
        src["mesh_cleanup"] = {k: rep.get(k) for k in ("backend", "output", "time_s")}
        shape = fc_core.mesh_to_solid(fc_mesh(v, f))
    elif a.get("file"):
        shape, src = fc_core.import_file(a["file"]), {"file": a["file"]}
    elif a.get("script"):
        shape, src = fc_core.run_script(a["script"], params), {"script": a["script"]}
    elif a.get("ops"):
        shape, src = fc_core.build_from_ops(a["ops"], params), {"ops": a["ops"]}
    else:
        raise ValueError("give ops, script or file")
    if not shape.Solids:
        raise ValueError("result has no solid (open shell / 2D?)")
    meta = {"source": next(iter(src)), **src, "params": params, "volume_mm3": round(shape.Volume, 2),
            "bbox": fc_core.bbox_list(shape), "faces": len(shape.Faces)}
    if a.get("material"):
        meta["material"] = a["material"]
    fc_core.save_model(a["name"], shape, meta)
    return fc_core.summary(shape, a["name"], a.get("material"), faces=a.get("list_faces", True))


def act_inspect_model(a):
    shape, meta = fc_core.load_model(a["name"])
    out = fc_core.summary(shape, a["name"], a.get("material") or meta.get("material"), faces=a.get("list_faces", True),
                          max_faces=int(a.get("max_faces", 60)))
    for k in ("source", "params", "derived_from", "optimization"):
        if k in meta:
            out[k] = meta[k]
    if a.get("select") is not None:
        out["selected_faces"] = fc_core.select_faces(shape, a["select"])
    return out


def act_list_models(a):
    return {"models": fc_core.list_models(), "materials": {k: v["label"] for k, v in fc_core.MATERIALS.items()}}


def act_delete_model(a):
    for p in fc_core.model_paths(a["name"]):
        if os.path.exists(p):
            os.remove(p)
    return {"deleted": a["name"]}


def act_export_model(a):
    shape, meta = fc_core.load_model(a["name"])
    fmt = a.get("format", "step").lower()
    path = a.get("path") or os.path.join(fc_core.EXPORT_DIR, f"{a['name']}.{ {'fcstd': 'FCStd'}.get(fmt, fmt) }")
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt in ("step", "stp"):
        shape.exportStep(path)
    elif fmt in ("iges", "igs"):
        shape.exportIges(path)
    elif fmt == "brep":
        shape.exportBrep(path)
    elif fmt == "stl":
        import Mesh
        import MeshPart
        m = MeshPart.meshFromShape(Shape=shape, LinearDeflection=float(a.get("deflection", 0.05)), AngularDeflection=0.3)
        m.write(path)
    elif fmt in ("glb", "gltf"):
        return act_export_web(dict(a, path=path if path.endswith(".glb") else path[: path.rfind(".")] + ".glb"))
    elif fmt == "fcstd":
        doc = App.newDocument("export")
        o = doc.addObject("Part::Feature", a["name"].replace("-", "_"))
        o.Shape = shape
        doc.recompute()
        doc.saveAs(path)
        App.closeDocument(doc.Name)
    else:
        raise ValueError("format: step, iges, brep, stl, fcstd, glb")
    return {"path": path, "bytes": os.path.getsize(path)}


# --------------------------------------------------------------------------- FEM
def _strip(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


def act_analyze(a):
    import fem_ccx
    shape, meta = fc_core.load_model(a["name"])
    mat = a.get("material") or meta.get("material") or "steel"
    res = fem_ccx.run_static(shape, mat, a.get("fixed"), a.get("loads"), mesh_size=a.get("mesh_size"),
                             order=a.get("order", 2), label=a["name"])
    out = _strip(res)
    sf = a.get("safety_factor")
    if sf:
        out["required_safety_factor"] = sf
        out["passes"] = out["safety_factor"]["vs_" + a.get("stress_metric", "p99")] >= float(sf)
    if a.get("render", True):
        out["preview"] = render_ccx(shape, res, os.path.join(res["workdir"], "stress.png"))
    return out


def _ccx_node_xyz_vm(res):
    """Node coordinates + vM from the saved FCStd result (needed for coloured previews)."""
    doc = App.openDocument(res["fcstd"])
    r = [o for o in doc.Objects if o.isDerivedFrom("Fem::FemResultObject")][-1]
    fm = r.Mesh.FemMesh
    xyz = np.array([[fm.getNodeById(n).x, fm.getNodeById(n).y, fm.getNodeById(n).z] for n in r.NodeNumbers])
    vm = np.array(r.vonMises, float)
    App.closeDocument(doc.Name)
    return xyz, vm


def render_ccx(shape, res, path):
    try:
        xyz, vm = _ccx_node_xyz_vm(res)
        return render_shape(shape, path, xyz, vm, f"von Mises [MPa], max {vm.max():.1f}")
    except Exception as e:
        log(f"render failed: {e}")
        return None


# --------------------------------------------------------------------------- rendering
def tessellate(shape, max_len=None):
    """Triangles for previews: fast OCC tessellation, then long triangles split 1->4 so that
    per-triangle colours can show stress gradients (non-conforming, fine for drawing)."""
    bb = shape.BoundBox
    pts, tris = shape.tessellate(max(bb.DiagonalLength / 500.0, 0.005))
    v = np.array([[p.x, p.y, p.z] for p in pts], float)
    T = v[np.array(tris, int)]  # (n,3,3)
    L = max_len or bb.DiagonalLength / 60.0
    for _ in range(8):
        e = np.linalg.norm(T - np.roll(T, -1, axis=1), axis=2).max(1)
        big = e > L
        if not big.any() or len(T) > 120000:
            break
        B = T[big]
        a, b, c = B[:, 0], B[:, 1], B[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        T = np.concatenate([T[~big], np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                            np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
    return T.reshape(-1, 3), np.arange(len(T) * 3).reshape(-1, 3)


def render_mesh(verts, tris, path, pts=None, vals=None, title="", views=((25, -60), (25, 120), (90, -90))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import cKDTree
    tri_xyz = verts[tris]
    n = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.abs(n @ light)
    if vals is not None:
        _, idx = cKDTree(pts).query(tri_xyz.mean(1))
        v = vals[idx]
        vmax = np.percentile(vals, 99.5) or 1.0
        cmap = plt.get_cmap("turbo")
        cols = cmap(np.clip(v / vmax, 0, 1))[:, :3]
    else:
        cols = np.tile([0.62, 0.68, 0.78], (len(tris), 1))
    cols = np.clip(cols * shade[:, None], 0, 1)
    fig = plt.figure(figsize=(3.6 * len(views) + 0.8, 3.8), dpi=100)
    lo, hi = verts.min(0), verts.max(0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2
    for k, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri_xyz, facecolors=cols, edgecolor="none", linewidths=0))
        for i, f in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            f(ctr[i] - rad, ctr[i] + rad)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(el, az)
        ax.set_xlabel("x", fontsize=7, labelpad=-8)
        ax.set_ylabel("y", fontsize=7, labelpad=-8)
        ax.set_zlabel("z", fontsize=7, labelpad=-8)
        ax.tick_params(labelsize=5, pad=-2)
    if vals is not None:
        sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, vmax))
        fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.02, fraction=0.025)
    fig.subplots_adjust(left=0.0, right=0.93, bottom=0.02, top=0.92, wspace=0.0)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.savefig(path)
    plt.close(fig)
    return path


def render_shape(shape, path, pts=None, vals=None, title="", views=None):
    v, t = tessellate(shape)
    return render_mesh(v, t, path, pts, vals, title, **({"views": views} if views else {}))


def _section_views(ax):
    """Half sections keep coord <= mid along ax: look straight at the cut face, plus two obliques."""
    straight = {0: (0, 0), 1: (0, 90), 2: (90, -90)}[ax]
    return (straight, (25, -60), (25, 120))


def act_render(a):
    shape, meta = fc_core.load_model(a["name"])
    path = a.get("path") or os.path.join(fc_core.EXPORT_DIR, f"{a['name']}_preview.png")
    if a.get("result_fcstd"):
        xyz, vm = _ccx_node_xyz_vm({"fcstd": a["result_fcstd"]})
        return {"preview": render_shape(shape, path, xyz, vm, f"{a['name']} von Mises [MPa], max {vm.max():.1f}")}
    return {"preview": render_shape(shape, path, title=a["name"])}


# --------------------------------------------------------------------------- voxel tools
def _scan_fill(T, origin, h, n, a, eps):
    """Even-odd scanline fill: one ray along axis a through every voxel column centre. Each triangle
    is tested only against the columns inside its footprint; crossings toggle a parity array."""
    b, c = [i for i in range(3) if i != a]
    flip = np.zeros((n[b], n[c], n[a] + 1), np.int32)
    gb = (T[:, :, b] - origin[b] - eps[0]) / h - 0.5  # in column-index units
    gc = (T[:, :, c] - origin[c] - eps[1]) / h - 0.5
    j0 = np.clip(np.ceil(gb.min(1)), 0, n[b]).astype(np.int64)
    j1 = np.clip(np.floor(gb.max(1)), -1, n[b] - 1).astype(np.int64)
    k0 = np.clip(np.ceil(gc.min(1)), 0, n[c]).astype(np.int64)
    k1 = np.clip(np.floor(gc.max(1)), -1, n[c] - 1).astype(np.int64)
    nj, nk = np.maximum(j1 - j0 + 1, 0), np.maximum(k1 - k0 + 1, 0)
    cnt = nj * nk
    tri = np.flatnonzero(cnt)
    for ch in np.array_split(tri, max(1, int(cnt[tri].sum() // 4_000_000) + 1)):
        cc = cnt[ch]
        t = np.repeat(ch, cc)
        r = np.arange(cc.sum()) - np.repeat(np.cumsum(cc) - cc, cc)
        j = j0[t] + r // nk[t]
        k = k0[t] + r % nk[t]
        # 2D barycentric of the column centre (j, k) in the projected triangle
        x0, x1, x2 = gb[t, 0], gb[t, 1], gb[t, 2]
        y0, y1, y2 = gc[t, 0], gc[t, 1], gc[t, 2]
        det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        ok = np.abs(det) > 1e-12
        det = np.where(ok, det, 1.0)
        w1 = ((j - x0) * (y2 - y0) - (x2 - x0) * (k - y0)) / det
        w2 = ((x1 - x0) * (k - y0) - (j - x0) * (y1 - y0)) / det
        w0 = 1.0 - w1 - w2
        ok &= (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        t, j, k, w0, w1, w2 = t[ok], j[ok], k[ok], w0[ok], w1[ok], w2[ok]
        sa = w0 * T[t, 0, a] + w1 * T[t, 1, a] + w2 * T[t, 2, a]
        i = np.clip(np.ceil((sa - origin[a]) / h - 0.5), 0, n[a]).astype(np.int64)  # first centre above
        np.add.at(flip, (j, k, i), 1)
    inside = (np.cumsum(flip, axis=2)[:, :, :n[a]] & 1).astype(bool)
    return np.moveaxis(inside, [0, 1, 2], [b, c, a])


def voxelize(shape, resolution):
    """Voxel centres inside the shape. Fast path: scanline parity on the tessellation along x, y and
    z with a 2-of-3 vote (robust to small mesh holes), ~1 s for 20k triangles. Falls back to ray
    booleans when the voxel volume is far off the solid volume (open / broken shapes)."""
    bb = shape.BoundBox
    dims = np.array([bb.XLength, bb.YLength, bb.ZLength])
    lo = np.array([bb.XMin, bb.YMin, bb.ZMin])
    h = dims.max() / float(resolution)
    n = np.maximum(np.ceil(dims / h - 1e-6).astype(int), 1)
    origin = lo - (n * h - dims) / 2
    try:
        t = time.time()
        pts, tris = shape.tessellate(max(h / 20.0, 1e-3))
        v = np.array([[p.x, p.y, p.z] for p in pts], float)
        T = v[np.array(tris, int)]
        eps = (h * 1.234567e-4, h * 2.718281e-4)  # keep rays off edges/vertices that sit on the grid
        votes = sum(_scan_fill(T, origin, h, n, ax, eps).astype(np.int8) for ax in range(3))
        mask = votes >= 2
        vol = shape.Volume
        ratio = mask.sum() * h ** 3 / vol if vol > 0 else 0.0
        log(f"scanline voxelize {len(T)} tris {time.time() - t:.2f}s, voxel/solid volume {ratio:.3f}")
        if 0.5 < ratio < 2.0:  # broken parity (open mesh) shows up as streaks or an empty grid
            return origin, h, n, mask
        log("scanline volume mismatch; falling back to ray booleans")
    except Exception as ex:
        log(f"scanline voxelize failed ({ex}); falling back to ray booleans")
    return _voxelize_occ(shape, origin, h, n)


def _voxelize_occ(shape, origin, h, n):
    a = int(np.argmax(n))
    b, c = [i for i in range(3) if i != a]
    mask = np.zeros(n, bool)
    span = (origin[a] - h, origin[a] + (n[a] + 1) * h)
    centers_a = origin[a] + (np.arange(n[a]) + 0.5) * h
    for j in range(n[b]):
        for k in range(n[c]):
            p0 = np.zeros(3)
            p0[b] = origin[b] + (j + 0.5) * h
            p0[c] = origin[c] + (k + 0.5) * h
            p1 = p0.copy()
            p0[a], p1[a] = span
            col = np.zeros(n[a], bool)
            try:
                seg = shape.common(Part.makeLine(V(*p0), V(*p1)))
                for e in seg.Edges:
                    s0 = e.Vertexes[0].Point[a]
                    s1 = e.Vertexes[-1].Point[a]
                    lo_, hi_ = min(s0, s1), max(s0, s1)
                    col |= (centers_a >= lo_) & (centers_a <= hi_)
            except Exception:
                for i, ca in enumerate(centers_a):
                    q = p0.copy()
                    q[a] = ca
                    col[i] = shape.isInside(V(*q), 1e-6, True)
            idx = [slice(None)] * 3
            idx[b], idx[c] = j, k
            mask[tuple(idx)] = col
    return origin, h, n, mask


def _tri_subcentroids(k):
    """Barycentric centroids of the k*k congruent sub-triangles of a triangle split k times per edge."""
    out = []
    for i in range(k):
        for j in range(k - i):
            out.append([i + 1 / 3, j + 1 / 3, k - i - j - 2 / 3])  # upright
            if i + j < k - 1:
                out.append([i + 2 / 3, j + 2 / 3, k - i - j - 4 / 3])  # inverted
    return np.array(out) / k


class BCMapper:
    """Maps FreeCAD face selections onto voxel nodes / elements."""

    def __init__(self, shape, grid, elems):
        from scipy.spatial import cKDTree
        self.shape, self.g, self.elems = shape, grid, elems
        self.faces = shape.Faces  # shape.Faces rebuilds the whole list on every access
        self.nodes = np.unique(grid.elem_nodes(elems))
        self.node_xyz = grid.node_xyz(self.nodes)
        self.tree = cKDTree(self.node_xyz)
        self.etree = cKDTree(grid.elem_centers(elems))

    def samples(self, ids):
        pts, w, nrm = [], [], []
        spacing = self.g.h / 2.0
        for i in ids:
            f = self.faces[i - 1]
            if len(f.Vertexes) == 3 and f.Surface.TypeId == "Part::GeomPlane":
                # mesh triangle: k x k equal sub-triangles analytically (no per-face tessellation)
                P = np.array([[q.X, q.Y, q.Z] for q in f.Vertexes])
                k = max(1, int(np.ceil(max(e.Length for e in f.Edges) / spacing)))
                c = _tri_subcentroids(k)  # (k*k, 3) barycentric
                nn = f.normalAt(0, 0)
                pts.append(c @ P)
                w.append(np.full(len(c), f.Area / len(c)))
                nrm.append(np.tile([nn.x, nn.y, nn.z], (len(c), 1)))
                continue
            p, ww = fc_core.face_samples(f, spacing)
            pts.append(p)
            w.append(ww)
            ns = []
            for q in p:
                try:
                    u, v = f.Surface.parameter(V(*q))
                    nn = f.normalAt(u, v)
                    ns.append([nn.x, nn.y, nn.z])
                except Exception:
                    ns.append([0, 0, 0])
            nrm.append(np.array(ns))
        return np.vstack(pts), np.concatenate(w), np.vstack(nrm)

    def surface_nodes(self, pts):
        _, near = self.tree.query(pts)
        within = self.tree.query_ball_point(pts, 0.5 * self.g.h)
        s = set(self.nodes[near].tolist())
        for lst in within:
            s.update(self.nodes[lst].tolist())
        return np.array(sorted(s))

    def load_nodes(self, pts, vecs):
        _, near = self.tree.query(pts)
        lc = {}
        for i, f in zip(near, vecs):
            gn = int(self.nodes[i])
            lc[gn] = lc.get(gn, 0) + f
        return lc

    def elems_near(self, pts, dist):
        hits = self.etree.query_ball_point(pts, max(dist, 1e-9))
        out = set()
        for lst in hits:
            out.update(lst)
        return np.array(sorted(out), int)


def _elems_with_nodes(grid, elems, node_ids):
    en = grid.elem_nodes(elems)
    return np.isin(en, node_ids).any(1)


def _normalize_fixed(fixed):
    return [fx if isinstance(fx, dict) and "faces" in fx else {"faces": fx} for fx in (fixed or [])]


def _loadcases(a):
    if a.get("load_cases"):
        return a["load_cases"]
    return [a.get("loads") or []]


def build_voxel_problem(shape, a, resolution):
    import topopt
    origin, h, n, mask = voxelize(shape, resolution)
    grid = topopt.Grid(origin, h, n)
    # voxels joined to the part only by an edge/corner are zero-stiffness hinges (singular K, CG
    # never converges): keep the face-connected components only, dropping specks < 1% of the largest
    from scipy import ndimage
    lab, k = ndimage.label(mask)
    if k > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = sizes >= max(0.01 * sizes.max(), 1)
        keep[0] = False
        dropped = int(mask.sum() - sizes[keep].sum())
        if dropped:
            mask = keep[lab]
            log(f"dropped {dropped} voxels in {int(k - (keep[1:]).sum())} specks not face-connected to the part")
    elems = np.argwhere(mask)
    if len(elems) == 0:
        raise ValueError("voxelization empty - raise resolution")
    if len(elems) > int(a.get("max_elements", 250000)):
        raise ValueError(f"{len(elems)} voxels > max_elements; lower resolution")
    bc = BCMapper(shape, grid, elems)
    keep_t = float(a.get("keep_thickness", h))
    passive = np.zeros(len(elems), bool)
    fixed = []
    for fx in _normalize_fixed(a.get("fixed")):
        ids = fc_core.select_faces(shape, fx["faces"])
        pts, _, _ = bc.samples(ids)
        nodes = bc.surface_nodes(pts)
        fixed.append((nodes, (fx.get("dofs") or "xyz").lower()))
        passive |= _elems_with_nodes(grid, elems, nodes)
        passive[bc.elems_near(pts, keep_t)] = True
    loadcases = []
    for case in _loadcases(a):
        lc = {}
        for ld in case:
            ids = fc_core.select_faces(shape, ld["faces"])
            pts, w, nrm = bc.samples(ids)
            if "pressure" in ld:
                vecs = -float(ld["pressure"]) * nrm * w[:, None] * (-1 if ld.get("reversed") else 1)
            else:
                f = np.array(ld["force"], float)
                vecs = f[None, :] * (w / w.sum())[:, None]
            for gn, f in bc.load_nodes(pts, vecs).items():
                lc[gn] = lc.get(gn, 0) + f
            passive |= _elems_with_nodes(grid, elems, np.array(list(bc.load_nodes(pts, vecs).keys())))
            passive[bc.elems_near(pts, keep_t)] = True
        loadcases.append(lc)
    for kp in a.get("keep") or []:
        if isinstance(kp, dict) and "region" in kp:
            (x0, y0, z0), (x1, y1, z1) = kp["region"]
            c = grid.elem_centers(elems)
            passive |= (c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1) & (c[:, 2] >= z0) & (c[:, 2] <= z1)
            continue
        sel = kp["faces"] if isinstance(kp, dict) and "faces" in kp else kp
        t = float(kp.get("thickness", keep_t)) if isinstance(kp, dict) else keep_t
        pts, _, _ = bc.samples(fc_core.select_faces(shape, sel))
        passive[bc.elems_near(pts, max(t, 0.5 * h))] = True
    if a.get("shell"):
        # print skin: every voxel within `shell` mm of any boundary (outer faces and holes) stays solid,
        # so only the interior is optimised and the visible surface is untouched
        from scipy import ndimage
        d = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1, 1:-1]
        depth = (d[elems[:, 0], elems[:, 1], elems[:, 2]] - 0.5) * h
        passive |= depth <= max(float(a["shell"]), 0.5 * h)
    return grid, elems, passive, fixed, loadcases


def import_skimage_measure():
    """FreeCAD's Ext/lazy_loader shadows the pip `lazy_loader` that scikit-image needs: swap it briefly."""
    if "skimage.measure" in sys.modules:
        return sys.modules["skimage.measure"]
    saved = {k: v for k, v in sys.modules.items() if k == "lazy_loader" or k.startswith("lazy_loader.")}
    for k in saved:
        del sys.modules[k]
    old_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if not p.rstrip("/").endswith("/Ext")]
    try:
        from skimage import measure
    finally:
        sys.path[:] = old_path
        for k in [k for k in sys.modules if k == "lazy_loader" or k.startswith("lazy_loader.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return measure


# --------------------------------------------------------------------------- mesh optimisation (CuMesh)
def mesh_optimize(verts, faces, target_faces=None, backend=None, device=None, **opts):
    """Repair + decimate a triangle mesh. GPU (CuMesh in the torch helper) first, FreeCAD CPU fallback.
    Returns (verts, faces, report)."""
    import topopt
    device = (device or os.environ.get("FREECAD_MCP_DEVICE", "auto")).lower()
    if device != "cpu":
        own = backend is None
        try:
            b = backend or topopt.GPUBackend()
            try:
                v, f, rep = b.mesh_optimize(verts, faces, target_faces=target_faces, **opts)
                rep["backend"] = f"gpu:cumesh ({b.info.get('device')})"
                return v, orient_outward(v, f, rep), rep
            finally:
                if own:
                    b.close()
        except Exception as ex:
            if device == "gpu":
                raise RuntimeError(f"GPU mesh optimisation failed: {ex}")
            log(f"CuMesh unavailable ({str(ex)[:160]}); CPU mesh fallback")
            note = f"CuMesh unavailable: {str(ex)[:160]}"
    else:
        note = None
    import Mesh
    t0 = time.time()
    m = Mesh.Mesh([[list(map(float, verts[i])) for i in tri] for tri in faces])
    rep = {"input": {"vertices": len(verts), "faces": len(faces)}, "backend": "cpu:freecad"}
    if note:
        rep["note"] = note
    m.removeDuplicatedPoints()
    m.removeDuplicatedFacets()
    m.fixDegenerations(1e-9)
    m.removeNonManifolds()
    m.harmonizeNormals()
    if opts.get("fill_holes"):
        m.fillupHoles(1000)
    if target_faces and m.CountFacets > target_faces:
        bb = m.BoundBox
        m.decimate(bb.DiagonalLength * 1e-3, 1.0 - target_faces / m.CountFacets)
    pts, tris = m.Topology
    v = np.array([[p.x, p.y, p.z] for p in pts], float)
    f = np.array(tris, int)
    rep["output"] = {"vertices": len(v), "faces": len(f)}
    rep["time_s"] = round(time.time() - t0, 3)
    return v, orient_outward(v, f, rep), rep


def signed_volume(verts, faces):
    t = np.asarray(verts, float)[np.asarray(faces, int)]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def orient_outward(verts, faces, rep=None):
    """Closed meshes must wind outward for FreeCAD solids/booleans; flip all faces if inside-out."""
    faces = np.asarray(faces)
    if signed_volume(verts, faces) < 0:
        faces = faces[:, [0, 2, 1]]
        if rep is not None:
            rep["flipped_winding"] = True
    return faces


def separate_coincident(verts, faces, eps, rep=None):
    """CuMesh can leave distinct vertices at identical positions where two sheets touch; FreeCAD
    would weld them into non-manifold edges. Push each copy `eps` inward along its own normal."""
    v = np.asarray(verts, float).copy()
    f = np.asarray(faces, int)
    _, inv, cnt = np.unique(np.round(v / max(eps, 1e-12), 0), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    dup = np.flatnonzero(cnt[inv] > 1)
    if len(dup) == 0:
        return v
    t = v[f]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30
    v[dup] -= eps * 2.0 * vn[dup]
    if rep is not None:
        rep["separated_coincident_vertices"] = int(len(dup))
    return v


def load_mesh_file(path):
    import trimesh
    tm = trimesh.load(os.path.expanduser(path), force="mesh", process=False)
    tm.merge_vertices()  # STL stores 3 unshared vertices per triangle
    return np.asarray(tm.vertices, float), np.asarray(tm.faces, int)


def fc_mesh(verts, faces):
    import Mesh
    return Mesh.Mesh([[list(map(float, verts[i])) for i in tri] for tri in faces])


def mesh_quality(verts, faces):
    try:
        import trimesh
        tm = trimesh.Trimesh(verts, faces, process=False)
        return {"watertight": bool(tm.is_watertight), "winding_consistent": bool(tm.is_winding_consistent),
                "volume_mm3": round(float(tm.volume), 2) if tm.is_watertight else None,
                "area_mm2": round(float(tm.area), 2), "bodies": int(tm.body_count)}
    except Exception as ex:
        return {"error": str(ex)[:120]}


def act_optimize_mesh(a):
    """3D model optimiser: repair + decimate a model or mesh file on the GPU (CuMesh)."""
    t0 = time.time()
    if a.get("file"):
        verts, faces = load_mesh_file(a["file"])
        src = {"file": a["file"]}
    else:
        shape, meta = fc_core.load_model(a["name"])
        bb = shape.BoundBox
        pts, tris = shape.tessellate(float(a.get("deflection") or bb.DiagonalLength / 1000.0))
        verts = np.array([[p.x, p.y, p.z] for p in pts], float)
        faces = np.array(tris, int)
        src = {"model": a["name"]}
    before = mesh_quality(verts, faces)
    target = a.get("target_faces")
    if target is None and a.get("reduction"):
        target = int(len(faces) * (1.0 - float(a["reduction"])))
    v, f, rep = mesh_optimize(verts, faces, target_faces=target, device=a.get("device"),
                              **{k: a[k] for k in ("fill_holes", "min_component_area", "repair") if a.get(k) is not None})
    after = mesh_quality(v, f)
    out_name = a.get("out_name") or (f"{a['name']}_mesh" if a.get("name") else
                                     os.path.splitext(os.path.basename(a["file"]))[0] + "_opt")
    stl = os.path.join(fc_core.EXPORT_DIR, out_name + ".stl")
    m = fc_mesh(v, f)
    m.write(stl)
    out = {"source": src, "out_name": out_name, "stl": stl, "before": before, "after": after, "cumesh": rep}
    if a.get("export_glb", True):
        g = _auto_glb(v, f, out_name)
        if g:
            out["glb"] = {k: g[k] for k in ("glb", "raw_bytes", "packed_bytes", "size_ratio")}
    if a.get("to_solid", True):
        if after.get("watertight"):
            try:
                solid = fc_core.mesh_to_solid(m)
                fc_core.save_model(out_name, solid, {"source": "optimize_mesh", **src, "volume_mm3": round(solid.Volume, 2),
                                                     "bbox": fc_core.bbox_list(solid), "faces": len(solid.Faces)})
                out["model"] = out_name
                out["solid_volume_mm3"] = round(solid.Volume, 2)
            except Exception as ex:
                out["solid_error"] = str(ex)[:200]
        else:
            out["solid_note"] = "mesh is not watertight - stored as STL only (try fill_holes=<max hole perimeter mm>)"
    if a.get("render", True):
        try:
            out["preview"] = render_mesh(v, f, os.path.join(fc_core.EXPORT_DIR, out_name + "_preview.png"),
                                         title=f"{out_name}: {before.get('area_mm2')}->{after.get('area_mm2')} mm2, "
                                               f"{len(faces)}->{len(f)} faces")
        except Exception as ex:
            log(f"preview failed {ex}")
    out["time_s"] = round(time.time() - t0, 2)
    return out


# --------------------------------------------------------------------------- glTF / gltfpack
def gltfpack_bin():
    import shutil
    for c in (os.environ.get("GLTFPACK"), shutil.which("gltfpack"), os.path.expanduser("~/.local/bin/gltfpack"),
              "/usr/local/bin/gltfpack"):
        if c and os.path.exists(c):
            return c
    raise RuntimeError("gltfpack not found (npm install -g gltfpack, or set GLTFPACK)")


def stress_colors(verts, xyz, vm):
    """Per-vertex turbo colours (RGBA uint8) of the nearest FEM node's von Mises stress."""
    import matplotlib
    from scipy.spatial import cKDTree
    _, idx = cKDTree(xyz).query(verts)
    vmax = float(np.percentile(vm, 99.5)) or 1.0
    rgba = matplotlib.colormaps["turbo"](np.clip(vm[idx] / vmax, 0, 1))
    return (rgba * 255).astype(np.uint8), vmax


def pack_glb(verts, faces, out_path, colors=None, scale=1e-3, compress="cc", simplify=None, simplify_error=None,
             aggressive=False, quantize=True, position_bits=None, extra_args=None, name="part"):
    """Write verts/faces (mm) as glTF (metres by default) with trimesh, then optimise with gltfpack:
    quantisation (KHR_mesh_quantization), meshopt compression (compress: none|c|cc|fallback) and
    optional simplification (simplify = triangle ratio 0..1). Returns a report dict."""
    import subprocess
    import trimesh
    t0 = time.time()
    tm = trimesh.Trimesh(np.asarray(verts, float) * scale, np.asarray(faces, int), process=False)
    if colors is not None:
        tm.visual = trimesh.visual.ColorVisuals(tm, vertex_colors=colors)
    raw = out_path[:-4] + "_raw.glb"
    scene = trimesh.Scene()
    scene.add_geometry(tm, node_name=name, geom_name=name)
    scene.export(raw)
    args = [gltfpack_bin(), "-i", raw, "-o", out_path]
    compress = (compress or "none").lower()
    args += {"c": ["-c"], "cc": ["-cc"], "fallback": ["-cf"], "none": []}.get(compress, ["-cc"])
    if simplify is not None and float(simplify) < 1:
        args += ["-si", str(float(simplify))]
        if simplify_error is not None:
            args += ["-se", str(float(simplify_error))]
        if aggressive:
            args += ["-sa"]
    if not quantize:
        args += ["-noq"]
    elif position_bits:
        args += ["-vp", str(int(position_bits))]
    report_json = out_path[:-4] + "_gltfpack.json"
    args += ["-r", report_json] + list(extra_args or [])
    p = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"gltfpack failed ({p.returncode}): {(p.stderr or p.stdout)[-400:]}")
    out = {"glb": out_path, "raw_glb": raw, "raw_bytes": os.path.getsize(raw), "packed_bytes": os.path.getsize(out_path),
           "triangles_in": int(len(faces)), "units": "m" if scale == 1e-3 else f"scale {scale}",
           "compression": compress, "quantized": bool(quantize), "time_s": round(time.time() - t0, 2)}
    out["size_ratio"] = round(out["packed_bytes"] / max(out["raw_bytes"], 1), 4)
    try:
        rep = json.load(open(report_json))
        tri = rep.get("render", {}).get("triangleCount") or rep.get("triangleCount")
        if tri is not None:
            out["triangles_out"] = tri
        out["gltfpack_report"] = report_json
    except Exception:
        pass
    if p.stderr.strip():
        out["gltfpack_log"] = p.stderr.strip()[-300:]
    return out


def minify_mesh(verts, faces, max_error_mm, name="part"):
    """Error-bounded simplification (meshoptimizer via gltfpack -se): collapses only where the surface
    moves less than max_error_mm, so flat / gently curved regions lose most triangles while detail
    stays. Returns (verts, faces) in the input units (mm)."""
    import subprocess
    import tempfile
    import trimesh
    v = np.asarray(verts, float)
    f = np.asarray(faces, int)
    if len(f) == 0 or max_error_mm <= 0:
        return v, f
    extent = float((v.max(0) - v.min(0)).max())
    with tempfile.TemporaryDirectory(dir=fc_core.RUN_DIR) as td:
        raw, out = os.path.join(td, "in.glb"), os.path.join(td, "out.glb")
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.Trimesh(v, f, process=False), node_name=name, geom_name=name)
        scene.export(raw)
        args = [gltfpack_bin(), "-i", raw, "-o", out, "-noq", "-si", "0.001", "-se", f"{max_error_mm / extent:.8f}"]
        p = subprocess.run(args, capture_output=True, text=True, timeout=600)
        if p.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"gltfpack simplify failed ({p.returncode}): {(p.stderr or p.stdout)[-300:]}")
        m = trimesh.load(out, force="mesh", process=False)
    m.merge_vertices()
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def act_export_web(a):
    """Web/AR-ready .glb: tessellate, optional CuMesh pre-decimation, optional stress colours, gltfpack."""
    shape, meta = fc_core.load_model(a["name"])
    bb = shape.BoundBox
    if a.get("result_fcstd") and len(shape.Faces) <= 500:
        # vertex colours need small triangles: bounded edge length instead of pure deflection
        import MeshPart
        m = MeshPart.meshFromShape(Shape=shape, MaxLength=float(a.get("max_edge") or bb.DiagonalLength / 80.0))
        pts, tris = m.Topology
    else:
        pts, tris = shape.tessellate(float(a.get("deflection") or bb.DiagonalLength / 2000.0))
    v = np.array([[p.x, p.y, p.z] for p in pts], float)
    f = np.array(tris, int)
    out = {"model": a["name"], "tessellation_faces": int(len(f))}
    if a.get("target_faces") and len(f) > int(a["target_faces"]):
        v, f, rep = mesh_optimize(v, f, target_faces=int(a["target_faces"]), device=a.get("device"))
        out["cumesh"] = {k: rep.get(k) for k in ("backend", "output", "time_s", "note") if rep.get(k)}
    colors = None
    if a.get("result_fcstd"):
        xyz, vm = _ccx_node_xyz_vm({"fcstd": a["result_fcstd"]})
        colors, vmax = stress_colors(v, xyz, vm)
        out["colored_by"] = {"von_mises_MPa_max_scale": round(vmax, 2), "colormap": "turbo"}
    path = os.path.expanduser(a.get("path") or os.path.join(fc_core.EXPORT_DIR, f"{a['name']}.glb"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out["gltfpack"] = pack_glb(v, f, path, colors=colors, scale=1e-3 if a.get("units", "m") == "m" else 1.0,
                               compress=a.get("compress", "cc"), simplify=a.get("simplify"),
                               simplify_error=a.get("simplify_error"), aggressive=a.get("aggressive", False),
                               quantize=a.get("quantize", True), position_bits=a.get("position_bits"),
                               name=a["name"])
    return out


def _auto_glb(verts, faces, name, log=log):
    """Packed .glb next to the other exports (silently skipped if gltfpack is missing)."""
    try:
        return pack_glb(verts, faces, os.path.join(fc_core.EXPORT_DIR, name + ".glb"), name=name)
    except Exception as ex:
        log(f"glb export skipped: {str(ex)[:160]}")
        return None


# --------------------------------------------------------------------------- geometry from densities
def _taubin(verts, faces, iters=10, lam=0.5, mu=-0.53):
    import scipy.sparse as sp
    nv = len(verts)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(nv, nv)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    W = sp.diags(1.0 / deg) @ A
    v = verts.copy()
    for _ in range(iters):
        v = v + lam * (W @ v - v)
        v = v + mu * (W @ v - v)
    return v


def density_to_shape(shape, grid, elems, field, level, smooth=10, target_faces=8000, log=log, backend=None,
                     device=None):
    """Marching cubes of the element density field, extended 2 voxels past the original boundary,
    then intersected with the original shape so original faces (supports, holes...) stay exact."""
    from scipy import ndimage
    import Mesh
    measure = import_skimage_measure()
    n = grid.n
    pad = 3
    F = np.zeros(tuple(np.array(n) + 2 * pad), float)
    dom = np.zeros_like(F, bool)
    e = elems + pad
    F[e[:, 0], e[:, 1], e[:, 2]] = field
    dom[e[:, 0], e[:, 1], e[:, 2]] = True
    # extend densities outward so that boolean-trimming with the original shape gives its exact faces
    dist, (ix, iy, iz) = ndimage.distance_transform_edt(~dom, return_indices=True)
    ext = F[ix, iy, iz]
    F = np.where(dom, F, np.where(dist <= 2.0, ext, 0.0))
    F = ndimage.gaussian_filter(F, 0.5)
    verts, faces, _, _ = measure.marching_cubes(F, level=level, spacing=(grid.h,) * 3)
    verts = verts + grid.o + grid.h * (0.5 - pad)
    if smooth:
        verts = _taubin(verts, faces, iters=int(smooth))
    n_mc = len(faces)
    mc_verts, mc_faces = verts, faces

    def to_solid(v, f, rep):
        if rep.get("backend", "").startswith("gpu"):
            v = separate_coincident(v, f, grid.h * 1e-3, rep)
        m = fc_mesh(v, f)
        m.harmonizeNormals()
        m.fixDegenerations(1e-6)
        m.removeDuplicatedPoints()
        m.removeDuplicatedFacets()
        return m, fc_core.mesh_to_solid(m)

    verts, faces, rep = mesh_optimize(mc_verts, mc_faces, target_faces=target_faces, backend=backend, device=device)
    log(f"mesh {n_mc} -> {len(faces)} faces via {rep.get('backend')} in {rep.get('time_s')}s")
    m, raw = to_solid(verts, faces, rep)
    if not raw.isValid() and rep.get("backend", "").startswith("gpu"):
        log("GPU-decimated surface gives an invalid solid; CPU decimation fallback")
        if os.environ.get("FREECAD_MCP_DEBUG"):
            np.savez(os.path.join(fc_core.RUN_DIR, "gpu_invalid_mesh.npz"), v=verts, f=faces, mv=mc_verts, mf=mc_faces)
        verts, faces, rep2 = mesh_optimize(mc_verts, mc_faces, target_faces=target_faces, device="cpu")
        rep2["note"] = "GPU surface invalid as solid, CPU decimation used"
        rep = rep2
        m, raw = to_solid(verts, faces, rep)
    log(f"mesh {m.CountFacets} facets, solid={m.isSolid()} valid={raw.isValid()}")
    try:
        t = time.time()
        cut = raw.common(shape)
        solids = sorted(cut.Solids, key=lambda s: -s.Volume)
        if not solids:
            raise RuntimeError("empty common")
        # slivers from the trim are dropped; real separate bodies (> 1% volume) are kept
        solids = [s for s in solids if s.Volume > 0.01 * solids[0].Volume]
        res = solids[0] if len(solids) == 1 else solids[0].fuse(solids[1:])
        log(f"boolean trim {time.time() - t:.1f}s")
        return res, m, "trimmed", rep
    except Exception as ex:
        log(f"boolean trim failed ({ex}); returning raw mesh solid")
        if os.environ.get("FREECAD_MCP_DEBUG"):
            np.savez(os.path.join(fc_core.RUN_DIR, "trim_fail_mesh.npz"), v=verts, f=faces)
        return raw, m, "mesh", rep


def voxelize_tris(T, resolution):
    """Scanline voxelization (2-of-3 vote) of a raw triangle array (n,3,3) on its own bbox grid."""
    lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
    dims = hi - lo
    h = dims.max() / float(resolution)
    n = np.maximum(np.ceil(dims / h - 1e-6).astype(int), 1)
    origin = lo - (n * h - dims) / 2
    eps = (h * 1.234567e-4, h * 2.718281e-4)
    votes = sum(_scan_fill(T, origin, h, n, ax, eps).astype(np.int8) for ax in range(3))
    return origin, h, n, votes >= 2


def cavity_mesh_fine(shape, grid, elems, field, shell, level=0.5, smooth=4, tris=None):
    """Voids of the optimised density field as inner-shell triangles, cut on a finer grid than the
    FEM one: the coarse field is interpolated onto voxels of ~shell/2 and every voxel within `shell`
    mm of the surface is forced solid, so the skin is exact however coarse the optimisation ran.
    Everything outside the part counts as solid -> no outer surface is generated. Returned
    inward-facing (negative signed volume) to append to the untouched original surface mesh."""
    from scipy import ndimage
    measure = import_skimage_measure()
    bb = shape.BoundBox
    hf = min(grid.h, shell / 2.0)
    res_f = int(np.ceil(max(bb.XLength, bb.YLength, bb.ZLength) / hf))
    if tris is not None:  # skin measured from the full-resolution source surface, not the decimated one
        of, hf, nf, mask = voxelize_tris(tris, res_f)
    else:
        of, hf, nf, mask = voxelize(shape, res_f)
    # coarse element field, solid outside the part, sampled at the fine voxel centres
    pad = 1
    Fc = np.ones(tuple(np.array(grid.n) + 2 * pad), np.float32)
    e = elems + pad
    Fc[e[:, 0], e[:, 1], e[:, 2]] = field
    Fc = ndimage.gaussian_filter(Fc, 0.5)
    axes = [((of[i] + (np.arange(nf[i]) + 0.5) * hf) - grid.o[i]) / grid.h - 0.5 + pad for i in range(3)]
    F = np.ones(tuple(nf), np.float32)
    for ix in range(nf[0]):  # slab by slab keeps the coordinate arrays small
        cy, cz = np.meshgrid(axes[1], axes[2], indexing="ij")
        F[ix] = ndimage.map_coordinates(Fc, [np.full_like(cy, axes[0][ix]), cy, cz], order=1, mode="nearest")
    depth = (ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1, 1:-1] - 0.5) * hf
    F[~mask | (depth <= shell + hf)] = 1.0  # one-voxel margin: interpolation, smoothing, sharp features
    pad = 2
    F = np.pad(F, pad, constant_values=1.0)
    if F.min() >= level:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    verts, faces, _, _ = measure.marching_cubes(F, level=level, spacing=(hf,) * 3)
    verts = verts + of + hf * (0.5 - pad)
    if smooth:
        verts = _taubin(verts, faces, iters=int(smooth))
    t = verts[faces]
    if np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() > 0:
        faces = faces[:, ::-1]
    return verts, faces


def _signed_volume(verts, faces):
    t = verts[faces]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def _half_section(verts, faces, ax, mid):
    keep = verts[faces][:, :, ax].mean(1) <= mid
    return faces[keep]


def _vstats(vm, solid_vol, mat):
    return {"max": round(float(vm.max()), 2), "p99": round(float(np.percentile(vm, 99)), 2),
            "p95": round(float(np.percentile(vm, 95)), 2), "mean": round(float(vm.mean()), 2),
            "sf_p99": round(mat["yield"] / max(float(np.percentile(vm, 99)), 1e-12), 3),
            "sf_max": round(mat["yield"] / max(float(vm.max()), 1e-12), 3)}


def make_backend(device, n_elems):
    """GPU (torch ROCm/CUDA helper) for the voxel solves, or None for CPU CHOLMOD.
    device: auto (GPU above GPU_MIN_ELEMENTS voxels) | gpu | cpu; default from FREECAD_MCP_DEVICE."""
    import topopt
    device = (device or os.environ.get("FREECAD_MCP_DEVICE", "auto")).lower()
    if device == "cpu" or (device == "auto" and n_elems < int(os.environ.get("FREECAD_GPU_MIN_ELEMENTS", "4000"))):
        return None, None
    try:
        b = topopt.GPUBackend(tol=float(os.environ.get("FREECAD_GPU_TOL", "1e-6")))
        log(f"GPU solver: {b.info.get('device')} ({b.info.get('free_mb')} MB free)")
        return b, None
    except Exception as ex:
        if device == "gpu":
            raise RuntimeError(f"GPU requested but unavailable: {ex}")
        log(f"GPU unavailable ({ex}); using CPU")
        return None, f"GPU unavailable ({str(ex)[:120]}), used CPU"


def act_optimize_topology(a):
    import topopt
    t0 = time.time()
    shape, meta = fc_core.load_model(a["name"])
    mat = fc_core.get_material(a.get("material") or meta.get("material") or "steel")
    res = int(a.get("resolution", 50))
    grid, elems, passive, fixed, lcs = build_voxel_problem(shape, a, res)
    h = grid.h
    log(f"grid {grid.n} h={h:.3f}mm active={len(elems)} passive={passive.sum()} fixed_nodes={sum(len(f[0]) for f in fixed)}")
    backend, dev_note = make_backend(a.get("device"), len(elems))
    fem = topopt.VoxelFEM(grid, elems, mat["nu"], fixed, lcs, backend=backend)
    rmin = max(float(a.get("filter_radius", 1.5 * h)) / h, 1.1)
    sym = None
    if a.get("symmetry"):
        sym = topopt.mirror_map(grid, elems, [s.lower() for s in a["symmetry"]])
    metric = a.get("stress_metric", "p99")
    full = topopt.evaluate(grid, elems, np.ones(len(elems), bool), mat["nu"], mat["E"], fixed, lcs, backend)
    full_s = _vstats(full["vm"], None, mat)
    log(f"full design: vm p99={full_s['p99']} max={full_s['max']} disp={full['disp_max']:.4g}")
    anchor = passive.copy()
    # draft: the load-path layout settles in ~20 iterations; normal/fine converge it further
    q = {"draft": (25, 1e-3, 0.03), "normal": (40, 1e-4, 0.01), "fine": (60, 1e-5, 0.005)}
    q_iters, q_solve_tol, q_tol = q.get(str(a.get("quality", "draft")).lower(), q["draft"])
    iters = int(a.get("iterations", q_iters))
    shell = float(a["shell"]) if a.get("shell") else None
    pf = float(passive.mean())

    def total_vf(vf):
        # with a shell, volume_fraction means the share of the interior kept (like slicer infill %)
        return pf + vf * (1 - pf) if shell else vf

    def run(vf, x0=None):
        vf = total_vf(vf)
        r = topopt.simp(fem, passive, vf, penal=float(a.get("penal", 3.0)), rmin=rmin, max_iter=iters, x0=x0,
                        sym_maps=sym, log=None, solve_tol=q_solve_tol, tol=q_tol)
        solid = topopt.threshold_to_volume(r["xPhys"], passive, vf)
        solid = topopt.keep_connected(grid, elems, solid, anchor)
        ev = topopt.evaluate(grid, elems, solid, mat["nu"], mat["E"], fixed, lcs, backend)
        st = _vstats(ev["vm"], None, mat)
        log(f"vf={vf:.3f} it={r['iterations']} t={r['time_s']}s solid={solid.mean():.3f} vm p99={st['p99']} max={st['max']} disp={ev['disp_max']:.4g}")
        return r, solid, ev, st

    search = []
    sf = a.get("target_safety_factor")
    if sf:
        sf = float(sf)
        allow = mat["yield"] / sf
        if full_s[metric] > allow:
            log("full part already exceeds the allowable stress")
        lo, hi = float(a.get("min_volume_fraction", 0.0 if shell else 0.08)), 1.0
        best = None
        x0 = None
        vf = float(a.get("volume_fraction", 0.3 if shell else 0.4))
        for step in range(int(a.get("search_steps", 6))):
            r, solid, ev, st = run(vf, x0)
            ok = st[metric] <= allow and ev["lost_force"] < 1e-9
            search.append({"volume_fraction": round(vf, 4), "stress_" + metric: st[metric], "ok": bool(ok),
                           "disp_max_mm": round(ev["disp_max"], 5)})
            if ok:
                hi = vf
                best = (vf, r, solid, ev, st)
            else:
                lo = vf
            x0 = r["x"]
            vf = 0.5 * (lo + hi)
            if hi - lo < 0.02:
                break
        if best is None:  # nothing passed: take the stiffest tried (largest vf) and say so
            vf = hi if hi < 1.0 else 0.95
            r, solid, ev, st = run(vf, x0)
            best = (vf, r, solid, ev, st)
        vf, r, solid, ev, st = best
    else:
        vf = float(a.get("volume_fraction", 0.25 if shell else 0.35))
        r, solid, ev, st = run(vf)

    # ---- geometry
    field = np.where(solid, np.maximum(r["xPhys"], 0.55), np.minimum(r["xPhys"], 0.45))
    out_name = a.get("out_name") or f"{a['name']}_opt"
    src_file = meta.get("file") or ""
    hollow = None
    # final pass: error-bounded simplification of the exported mesh (mm; 0 / minify=False keeps every triangle)
    minify_err = float(a.get("minify_error", 0.05)) if a.get("minify", True) else 0.0
    if shell and os.path.splitext(src_file)[1].lower() in (".stl", ".obj", ".ply", ".off") and os.path.exists(src_file):
        # mesh source + skin: keep the ORIGINAL full-resolution triangles untouched and add the voids as
        # inner shells (no decimated rebuild, no boolean with the often invalid mesh solid)
        t = time.time()
        v0, f0 = load_mesh_file(src_file)
        cv, cf = cavity_mesh_fine(shape, grid, elems, field, shell, smooth=int(a.get("smooth", 8)) // 2,
                                  tris=v0[f0])
        vol0 = abs(_signed_volume(v0, f0))
        n_orig, n_cav = len(f0), len(cf)
        if minify_err:  # surface within minify_error, hidden cavities 2x (stays inside the 1-voxel skin margin)
            tm = time.time()
            v0, f0 = minify_mesh(v0, f0, minify_err)
            if len(cf):
                cv, cf = minify_mesh(cv, cf, 2 * minify_err)
            log(f"minify: surface {n_orig} -> {len(f0)}, cavities {n_cav} -> {len(cf)} triangles ({time.time() - tm:.1f}s)")
        vcav = _signed_volume(cv, cf)
        hollow = {"verts": np.vstack([v0, cv]), "faces": np.vstack([f0, cf + len(v0)]), "cav": (cv, cf),
                  "original_triangles": int(n_orig), "cavity_triangles": int(n_cav),
                  "output_triangles": int(len(f0) + len(cf)), "output_cavity_triangles": int(len(cf))}
        vol1 = vol0 + vcav
        new_shape, mesh, kind, mesh_rep = shape, None, "original_mesh_with_cavities", {}
        log(f"hollow mesh: {len(f0)} surface + {len(cf)} cavity triangles, cavities {-vcav:.0f} mm3 ({time.time() - t:.1f}s)")
    else:
        new_shape, mesh, kind, mesh_rep = density_to_shape(shape, grid, elems, field, 0.5, smooth=int(a.get("smooth", 8)),
                                                           target_faces=int(a.get("target_faces", 6000)),
                                                           backend=backend, device=a.get("device"))
        vol0, vol1 = shape.Volume, new_shape.Volume
    bcs = {"fixed": a.get("fixed"), "loads": a.get("loads"), "load_cases": a.get("load_cases")}
    fc_core.save_model(out_name, new_shape, {
        "source": "topology_optimization", "derived_from": a["name"], "geometry": kind,
        "material": a.get("material") or meta.get("material"), "volume_mm3": round(vol1, 2),
        "bbox": fc_core.bbox_list(new_shape), "faces": len(new_shape.Faces), "bcs_on_original": bcs})
    stl = os.path.join(fc_core.EXPORT_DIR, out_name + ".stl")
    if hollow:
        import trimesh
        trimesh.Trimesh(hollow["verts"], hollow["faces"], process=False).export(stl)
        glb = _auto_glb(hollow["verts"], hollow["faces"], out_name) if a.get("export_glb", True) else None
        step = None  # the hollowed part exists as a mesh only
    else:
        mesh_out = __import__("MeshPart").meshFromShape(Shape=new_shape, LinearDeflection=h * 0.05, AngularDeflection=0.3)
        mp, mt = mesh_out.Topology
        mv, mf = np.array([[q.x, q.y, q.z] for q in mp]), np.array(mt, int)
        if minify_err:
            try:
                mv, mf = minify_mesh(mv, mf, minify_err)
            except Exception as ex:
                log(f"minify skipped: {str(ex)[:160]}")
        import trimesh
        trimesh.Trimesh(mv, mf, process=False).export(stl)
        glb = _auto_glb(mv, mf, out_name) if a.get("export_glb", True) else None
        step = os.path.join(fc_core.EXPORT_DIR, out_name + ".step")
        try:
            new_shape.exportStep(step)
        except Exception:
            step = None

    out = {
        "model": out_name,
        "geometry": kind,
        "grid": {"elements": list(grid.n), "voxel_mm": round(h, 3), "active": int(len(elems)), "kept_passive": int(passive.sum()),
                 "filter_radius_mm": round(rmin * h, 3)},
        "material": {k: mat[k] for k in ("label", "E", "yield", "density")},
        "original": {"volume_mm3": round(vol0, 1), "mass_g": round(vol0 * mat["density"] / 1000, 2),
                     "voxel_fem": {"von_mises_MPa": full_s, "disp_max_mm": round(full["disp_max"], 5),
                                   "compliance": round(full["compliance"], 5)}},
        "optimized": {"volume_mm3": round(vol1, 1), "mass_g": round(vol1 * mat["density"] / 1000, 2),
                      "volume_fraction_target": round(total_vf(vf), 4),
                      "material_saved_pct": round(100 * (1 - vol1 / vol0), 1),
                      "voxel_fem": {"von_mises_MPa": st, "disp_max_mm": round(ev["disp_max"], 5),
                                    "compliance": round(ev["compliance"], 5),
                                    "stiffness_vs_original": round(full["compliance"] / max(ev["compliance"], 1e-30), 3)},
                      "simp_iterations": r["iterations"]},
        "exports": {"stl": stl, "step": step, "glb": glb and glb["glb"],
                    "glb_bytes": glb and glb["packed_bytes"]},
        "surface_mesh": {k: mesh_rep.get(k) for k in ("backend", "input", "output", "time_s", "note") if mesh_rep.get(k)},
        "note": ("voxel FEM numbers use the same voxel grid for both designs (compare them with each other); "
                 "stresses at load/support edges are singular - p99 is the robust metric. "
                 "Run verify (CalculiX) for final numbers."),
    }
    if shell:
        out["shell"] = {"thickness_mm": shell, "effective_mm": round(max(shell, h), 3),
                        "kept_fraction": round(pf, 4), "interior_fraction_kept": round(vf, 4),
                        "note": "outer surface is the original geometry; interior has closed voids - for resin "
                                "printing add drain holes; void ceilings may need <=45 deg overhang on FDM"}
        if hollow:
            out["shell"]["full_resolution"] = {
                "stl": stl, "original_triangles": hollow["original_triangles"],
                "cavity_triangles": hollow["cavity_triangles"], "output_triangles": hollow["output_triangles"],
                "output_cavity_triangles": hollow["output_cavity_triangles"],
                "minify_error_mm": minify_err or None,
                "note": ("exported STL = the source mesh simplified within minify_error (mean deviation ~1/3 of it) "
                         "+ cavity shells" if minify_err else "exported STL = the untouched source mesh + cavity shells")
                        + "; the stored model / STEP stay the (decimated) original - the hollow part exists as a mesh only"}
        if hollow:
            out["shell"]["effective_mm"] = shell  # the skin is cut on the fine geometry grid, exact
        elif shell < h:
            out["shell"]["warning"] = f"shell {shell} mm < voxel {h:.3f} mm: wall is 1 voxel; raise resolution for thinner walls"
    if search:
        out["search"] = {"target_safety_factor": sf, "allowable_MPa": round(mat["yield"] / sf, 2), "metric": metric,
                         "steps": search}
    if fem.lost_force > 0:
        out["warning"] = f"{fem.lost_force:.3g} N of load did not land on voxels"
    prev = os.path.join(fc_core.EXPORT_DIR, out_name + "_preview.png")
    try:
        cen = grid.elem_centers(elems[solid])
        vm, view, title = ev["vm"], new_shape, f"{out_name}: -{out['optimized']['material_saved_pct']}% material, voxel von Mises [MPa]"
        if hollow:  # half section: decimated original surface + the cavities
            bb = shape.BoundBox
            ax = int(np.argmin([bb.XLength, bb.YLength, bb.ZLength]))  # cut through the thinnest axis
            mid = [bb.XMin, bb.YMin, bb.ZMin][ax] + [bb.XLength, bb.YLength, bb.ZLength][ax] / 2
            pv, pf = tessellate(shape)
            cv, cf = hollow["cav"]
            vv = np.vstack([pv, cv])
            ff = np.vstack([_half_section(pv, pf, ax, mid), _half_section(cv, cf, ax, mid) + len(pv)])
            sel = cen[:, ax] <= mid
            out["preview"] = render_mesh(vv, ff, prev, cen[sel], vm[sel], title + " - half section", views=_section_views(ax))
            view = None
        elif shell:  # the skin hides everything: show a half section through the thinnest axis
            bb = new_shape.BoundBox
            ax = int(np.argmin([bb.XLength, bb.YLength, bb.ZLength]))  # cut through the thinnest axis
            lo = np.array([bb.XMin, bb.YMin, bb.ZMin]) - 1
            size = np.array([bb.XLength, bb.YLength, bb.ZLength]) + 2
            mid = lo[ax] + size[ax] / 2
            size[ax] /= 2
            view = new_shape.common(Part.makeBox(*map(float, size), V(*map(float, lo))))
            sel = cen[:, ax] <= mid
            cen, vm, title = cen[sel], vm[sel], title + " - half section"
        if view is not None:
            out["preview"] = render_shape(view, prev, cen, vm, title, views=_section_views(ax) if shell else None)
    except Exception as ex:
        log(f"preview failed {ex}")
    out["solver"] = dict(fem.solver_stats, time_s=round(fem.solver_stats["time_s"], 2))
    if dev_note:
        out["solver"]["note"] = dev_note
    fem.close()
    if backend is not None:
        backend.close()
    if a.get("verify", True):
        try:
            a.setdefault("verify_mesh_size", round(h, 3))
            out["verify_calculix"] = verify_ccx(shape, new_shape, a, mat, out_name)
        except Exception as ex:
            out["verify_calculix"] = {"error": str(ex)[:400]}
    out["time_s"] = round(time.time() - t0, 1)
    return out


def _map_selectors(orig_shape, new_shape, sel_list, key="faces", tol=0.02):
    mapped = []
    orig_faces = orig_shape.Faces
    for item in sel_list or []:
        item = dict(item) if isinstance(item, dict) and key in item else {key: item}
        ids = fc_core.select_faces(orig_shape, item[key])
        new_ids = fc_core.faces_on(new_shape, [orig_faces[i - 1] for i in ids], tol)
        item[key] = new_ids
        mapped.append(item)
    return mapped


def verify_ccx(orig, new_shape, a, mat, label):
    import fem_ccx
    fixed = _map_selectors(orig, new_shape, _normalize_fixed(a.get("fixed")))
    results = []
    for k, case in enumerate(_loadcases(a)):
        loads = _map_selectors(orig, new_shape, case)
        r = fem_ccx.run_static(new_shape, mat, fixed, loads, mesh_size=a.get("verify_mesh_size"), order=2,
                               label=f"{label}_verify{k}")
        r0 = None
        if a.get("verify_original", True):
            r0 = fem_ccx.run_static(orig, mat, _normalize_fixed(a.get("fixed")), case, mesh_size=a.get("verify_mesh_size"),
                                    order=2, label=f"{label}_orig{k}")
        entry = {"load_case": k, "optimized": {kk: _strip(r)[kk] for kk in ("von_mises_MPa", "displacement_mm", "safety_factor", "mesh", "fcstd")}}
        if r0:
            entry["original"] = {kk: _strip(r0)[kk] for kk in ("von_mises_MPa", "displacement_mm", "safety_factor")}
        if k == 0:
            entry["preview"] = render_ccx(new_shape, r, os.path.join(fc_core.EXPORT_DIR, f"{label}_ccx_stress.png"))
        results.append(entry)
    return results


# --------------------------------------------------------------------------- dimension optimisation
def act_optimize_dimensions(a):
    """Minimise mass of a parametric model (ops/script with params) s.t. CalculiX stress <= yield/SF."""
    import optuna
    import fem_ccx
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mat = fc_core.get_material(a.get("material") or "steel")
    sf = float(a.get("safety_factor", 2.0))
    allow = mat["yield"] / sf
    metric = a.get("stress_metric", "p99")
    bounds = a["bounds"]
    fixed_params = a.get("params") or {}
    max_disp = a.get("max_displacement")
    trials = []

    def build(p):
        pp = dict(fixed_params, **p)
        if a.get("script"):
            return fc_core.run_script(a["script"], pp)
        return fc_core.build_from_ops(a["ops"], pp)

    def objective(trial):
        p = {}
        for k, b in bounds.items():
            if isinstance(b, dict) and "choices" in b:
                p[k] = trial.suggest_categorical(k, b["choices"])
            elif isinstance(b, (list, tuple)) and len(b) == 3:
                p[k] = trial.suggest_float(k, float(b[0]), float(b[1]), step=float(b[2]))
            else:
                p[k] = trial.suggest_float(k, float(b[0]), float(b[1]))
        rec = {"params": p}
        try:
            s = build(p)
            r = fem_ccx.run_static(s, mat, a.get("fixed"), a.get("loads"), mesh_size=a.get("mesh_size"),
                                   order=a.get("order", 2), label=f"dimopt_{trial.number}", save_fcstd=False)
            stress = r["von_mises_MPa"][metric]
            disp = r["displacement_mm"]["max"]
            mass = s.Volume * mat["density"] / 1000
            viol = max(0.0, stress / allow - 1.0)
            if max_disp:
                viol += max(0.0, disp / float(max_disp) - 1.0)
            rec.update(mass_g=round(mass, 3), stress=round(stress, 2), disp=round(disp, 5), ok=viol == 0)
            trial.set_user_attr("viol", viol)
            trials.append(rec)
            log(f"trial {trial.number} {p} mass={mass:.2f} stress={stress:.1f} ok={viol == 0}")
            return mass * (1 + 10 * viol) + (1e3 * viol if viol > 0 else 0)
        except Exception as ex:
            rec.update(error=str(ex)[:200])
            trials.append(rec)
            trial.set_user_attr("viol", 1e9)
            return 1e12

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(
        seed=int(a.get("seed", 0)), constraints_func=lambda t: [t.user_attrs.get("viol", 1e9)]))
    if a.get("start"):
        study.enqueue_trial(a["start"])
    study.optimize(objective, n_trials=int(a.get("n_trials", 20)))
    ok = [t for t in trials if t.get("ok")]
    out = {"allowable_MPa": round(allow, 2), "metric": metric, "material": mat["label"], "n_trials": len(trials),
           "trials": sorted(trials, key=lambda t: (not t.get("ok", False), t.get("mass_g", 1e18)))[:int(a.get("show_trials", 8))]}
    if not ok:
        out["result"] = "no feasible design found - widen bounds or lower the safety factor"
        return out
    best = min(ok, key=lambda t: t["mass_g"])
    out["best"] = best
    if a.get("out_name"):
        s = build(best["params"])
        fc_core.save_model(a["out_name"], s, {"source": "dimension_optimization", "params": dict(fixed_params, **best["params"]),
                                              "ops": a.get("ops"), "script": a.get("script"), "material": a.get("material"),
                                              "volume_mm3": round(s.Volume, 2), "bbox": fc_core.bbox_list(s), "faces": len(s.Faces)})
        out["saved_model"] = a["out_name"]
    return out


ACTIONS = {k[4:]: v for k, v in globals().items() if k.startswith("act_")}


def main():
    job = json.load(open(sys.argv[1]))
    try:
        res = {"ok": True, "result": ACTIONS[job.pop("action")](job)}
    except Exception as e:
        log(traceback.format_exc())
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    with open(sys.argv[2], "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
