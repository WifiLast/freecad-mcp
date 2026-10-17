"""GPU mesh optimisation with CuMesh (ROCm build in third_party/CuMesh), used inside gpu_solver.py.

optimize(): repair (degenerate/duplicate faces, non-manifold edges, small islands, holes, winding)
and QEM-decimate a triangle mesh on the GPU. Returns numpy arrays plus a short report.
"""
import os
import sys
import time

import numpy as np
import torch

CUMESH_DIR = os.environ.get("FREECAD_CUMESH_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                               "third_party", "CuMesh"))
_cm = None


def cumesh():
    global _cm
    if _cm is None:
        if CUMESH_DIR not in sys.path:
            sys.path.insert(0, CUMESH_DIR)
        import cumesh as m
        _cm = m
    return _cm


def available():
    try:
        cumesh()
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _stats(m):
    return {"vertices": int(m.num_vertices), "faces": int(m.num_faces)}


def optimize(vertices, faces, target_faces=None, repair=True, fill_holes=None, min_component_area=None,
             unify_orientation=True, simplify_options=None, device="cuda"):
    """vertices (V,3) float, faces (F,3) int. fill_holes = max hole perimeter (model units) or None.
    min_component_area: drop islands smaller than this (default 0.1 % of the total area)."""
    cm = cumesh()
    t0 = time.time()
    v = torch.as_tensor(np.asarray(vertices, np.float32), device=device)
    f = torch.as_tensor(np.asarray(faces, np.int32), device=device)
    m = cm.CuMesh()
    m.init(v, f)
    rep = {"input": _stats(m), "steps": []}

    def step(name, fn, *args):
        try:
            fn(*args)
            rep["steps"].append({name: _stats(m)})
        except Exception as e:  # one failing repair must not lose the mesh
            rep["steps"].append({name: f"failed: {type(e).__name__}: {str(e)[:120]}"})

    if repair:
        step("remove_degenerate_faces", m.remove_degenerate_faces)
        step("remove_duplicate_faces", m.remove_duplicate_faces)
        step("repair_non_manifold_edges", m.repair_non_manifold_edges)
        if min_component_area is None:
            vv, ff = v.double(), f.long()
            area = torch.linalg.cross(vv[ff[:, 1]] - vv[ff[:, 0]], vv[ff[:, 2]] - vv[ff[:, 0]]).norm(dim=1).sum() / 2
            min_component_area = float(area) * 1e-3
        if min_component_area > 0:
            step("remove_small_connected_components", m.remove_small_connected_components, float(min_component_area))
        if fill_holes:
            step("fill_holes", m.fill_holes, float(fill_holes))
        if unify_orientation:
            step("unify_face_orientations", m.unify_face_orientations)
    if target_faces and m.num_faces > int(target_faces):
        step("simplify", m.simplify, int(target_faces), False, simplify_options or {})
        if repair:
            step("remove_degenerate_faces", m.remove_degenerate_faces)
            step("repair_non_manifold_edges", m.repair_non_manifold_edges)
    vo, fo = m.read()
    rep["output"] = _stats(m)
    try:
        rep["boundary_loops"] = int(m.num_boundary_loops)
    except Exception:
        pass
    if device != "cpu":
        torch.cuda.synchronize()
    rep["time_s"] = round(time.time() - t0, 3)
    return vo.cpu().numpy().astype(np.float64), fo.cpu().numpy().astype(np.int64), rep
