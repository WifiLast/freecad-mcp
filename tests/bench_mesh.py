"""CuMesh (GPU) vs FreeCAD (CPU) repair + decimation on a 330k-face sphere -> 6000 faces."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fc_worker as W  # FreeCAD init wipes earlier __main__ names
import time
import numpy as np, trimesh
s = trimesh.creation.icosphere(subdivisions=7, radius=20.0)
for dev in ["gpu", "cpu"]:
    t = time.time(); v, f, rep = W.mesh_optimize(s.vertices, s.faces, target_faces=6000, device=dev)
    q = W.mesh_quality(v, f)
    print(dev, len(s.faces), "->", len(f), "faces", round(time.time() - t, 2), "s (incl. transfer)", q["watertight"],
          round(q["volume_mm3"] / s.volume, 4) if q["volume_mm3"] else None)
