"""CuMesh ROCm smoke test: optimise a noisy marching-cubes sphere with duplicates via mesh_gpu."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, trimesh, mesh_gpu
ok, err = mesh_gpu.available(); print("cumesh import:", ok, err)
s = trimesh.creation.icosphere(subdivisions=6, radius=20.0)
v, f = s.vertices, s.faces
f = np.vstack([f, f[:500]])                       # duplicate faces
print("in", len(v), len(f))
t = time.time(); v2, f2, rep = mesh_gpu.optimize(v, f, target_faces=5000); print("time", round(time.time() - t, 2))
print(rep)
tm = trimesh.Trimesh(v2, f2, process=False); print("watertight", tm.is_watertight, "volume", round(tm.volume, 1), "ref", round(s.volume, 1))
