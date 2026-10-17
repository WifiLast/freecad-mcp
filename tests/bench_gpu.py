"""Compare CPU (CHOLMOD) and GPU (matrix-free PCG) voxel solves on a cantilever grid."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, topopt

def problem(nx, ny, nz):
    g = topopt.Grid((0, 0, 0), 1.0, (nx, ny, nz)); el = np.argwhere(np.ones(g.n, bool))
    fixed = [(np.unique(g.elem_nodes(el[el[:, 0] == 0])[:, [0, 3, 4, 7]]), "xyz")]
    ln = np.unique(g.elem_nodes(el[el[:, 0] == nx - 1])[:, [1, 2, 5, 6]])
    return g, el, fixed, [{int(n): np.array([0, 0, -1000.0 / len(ln)]) for n in ln}]

gpu = topopt.GPUBackend(); print(gpu.info)
for dims in [(40, 10, 14), (60, 20, 30), (100, 30, 40)]:
    g, el, fixed, lcs = problem(*dims)
    rng = np.random.default_rng(0); x = np.clip(0.3 + 0.2 * rng.standard_normal(len(el)), 1e-3, 1)
    Ee = 1e-6 + x ** 3
    res = {}
    for name, be in [("gpu", gpu), ("cpu", None)]:
        if name == "cpu" and len(el) > 80000: continue
        f = topopt.VoxelFEM(g, el, 0.3, fixed, lcs, backend=be)
        t = time.time(); f.solve(Ee); ce = f.elem_energy(); t1 = time.time() - t
        t = time.time(); f.solve(Ee * 1.01); t2 = time.time() - t   # warm start like a SIMP step
        res[name] = (round(t1, 2), round(t2, 2), f.compliance(), f.solver_stats.get("cg_iterations"))
        f.close()
    c = res.get("cpu", (0, 0, None))[2]
    print(dims, len(el), "elements", {k: v[:2] + (v[3],) for k, v in res.items()},
          "rel diff compliance", None if c is None else abs(res["gpu"][2] - c) / c)
gpu.close()
