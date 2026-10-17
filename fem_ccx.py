"""Static linear FEM with FreeCAD FEM + gmsh + CalculiX (ccx), headless."""
import os
import time

import numpy as np

import fc_core
from fc_core import App, Part, V

import ObjectsFem  # noqa: E402
from femtools import ccxtools  # noqa: E402
from femmesh.gmshtools import GmshTools  # noqa: E402

os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_WARNINGS", "0")


def run_static(shape, material, fixed, loads, mesh_size=None, order=2, workdir=None, label="fem",
               save_fcstd=True, keep_doc=False, matrix_solver=None):
    """fixed: list of {"faces": selector, "dofs": "xyz"} or plain selectors.
    loads: list of {"faces": selector, "force": [fx,fy,fz] N} | {"faces":..., "pressure": MPa}
    (force is total over the selected faces, any direction; pressure acts against the face normal).
    """
    t0 = time.time()
    mat = fc_core.get_material(material)
    workdir = workdir or os.path.join(fc_core.RUN_DIR, f"{label}_{int(time.time()*1000)}")
    os.makedirs(workdir, exist_ok=True)
    doc = App.newDocument("fem")
    body = doc.addObject("Part::Feature", "Body")
    body.Shape = shape
    an = ObjectsFem.makeAnalysis(doc, "Analysis")
    sol = ObjectsFem.makeSolverCalculiXCcxTools(doc, "Solver")
    sol.AnalysisType = "static"
    sol.MatrixSolverType = matrix_solver or os.environ.get("FREECAD_CCX_SOLVER", "default")
    try:
        sol.GeometricalNonlinearity = "linear"
    except Exception:
        pass
    an.addObject(sol)
    mo = ObjectsFem.makeMaterialSolid(doc, "Material")
    m = mo.Material
    m["Name"] = mat["label"]
    m["YoungsModulus"] = f"{mat['E']} MPa"
    m["PoissonRatio"] = str(mat["nu"])
    m["Density"] = f"{mat['density'] * 1000} kg/m^3"
    mo.Material = m
    an.addObject(mo)

    used = {"fixed": [], "loads": []}
    for k, fx in enumerate(fixed or []):
        if not (isinstance(fx, dict) and "faces" in fx):
            fx = {"faces": fx}
        ids = fc_core.select_faces(shape, fx["faces"])
        refs = [(body, f"Face{i}") for i in ids]
        dofs = (fx.get("dofs") or "xyz").lower()
        if dofs == "xyz":
            c = ObjectsFem.makeConstraintFixed(doc, f"Fixed{k}")
            c.References = refs
        else:
            c = ObjectsFem.makeConstraintDisplacement(doc, f"Disp{k}")
            c.References = refs
            for a in "xyz":
                if a in dofs:
                    setattr(c, f"{a}Free", False)
                    setattr(c, f"{a}Displacement", 0.0)
                else:
                    setattr(c, f"{a}Free", True)
        an.addObject(c)
        used["fixed"].append({"faces": ids, "dofs": dofs})
    if not used["fixed"]:
        raise ValueError("need at least one fixed support")

    for k, ld in enumerate(loads or []):
        ids = fc_core.select_faces(shape, ld["faces"])
        refs = [(body, f"Face{i}") for i in ids]
        if "pressure" in ld:
            c = ObjectsFem.makeConstraintPressure(doc, f"Pressure{k}")
            c.References = refs
            c.Pressure = f"{float(ld['pressure'])} MPa"
            c.Reversed = bool(ld.get("reversed", False))
            an.addObject(c)
            used["loads"].append({"faces": ids, "pressure_MPa": float(ld["pressure"])})
            continue
        f = np.array(ld["force"], float)
        if f.shape == ():
            raise ValueError("force must be a vector [fx, fy, fz] in N")
        mag = float(np.linalg.norm(f))
        if mag <= 0:
            continue
        c = ObjectsFem.makeConstraintForce(doc, f"Force{k}")
        c.References = refs
        c.Force = f"{mag} N"
        d = doc.addObject("Part::Feature", f"Dir{k}")
        d.Shape = Part.makeLine(V(0, 0, 0), V(*(f / mag)))
        d.Visibility = False
        c.Direction = (d, ["Edge1"])
        c.Reversed = False
        an.addObject(c)
        used["loads"].append({"faces": ids, "force_N": f.round(4).tolist()})

    bb = shape.BoundBox
    if not mesh_size:
        # ~ 3 elements over the thinnest bbox extent, capped by a global count heuristic
        dims = sorted([bb.XLength, bb.YLength, bb.ZLength])
        mesh_size = max(min(dims[0] / 3.0, (shape.Volume / 30000.0) ** (1 / 3.0) * 2.0), dims[2] / 200.0)
    mesh = ObjectsFem.makeMeshGmsh(doc, "Mesh")
    mesh.Shape = body
    mesh.CharacteristicLengthMax = f"{mesh_size} mm"
    mesh.CharacteristicLengthMin = f"{mesh_size / 4.0} mm"
    mesh.ElementOrder = "2nd" if int(order) == 2 else "1st"
    an.addObject(mesh)
    doc.recompute()
    err = GmshTools(mesh).create_mesh()
    if err:
        raise RuntimeError(f"gmsh failed: {err}")
    doc.recompute()
    n_nodes = mesh.FemMesh.NodeCount
    n_vol = mesh.FemMesh.VolumeCount
    if n_vol == 0:
        raise RuntimeError("gmsh produced no volume elements (open or invalid shape?)")
    t_mesh = time.time() - t0

    fea = ccxtools.FemToolsCcx(an, sol)
    fea.update_objects()
    fea.setup_working_dir(workdir)
    fea.setup_ccx()
    msg = fea.check_prerequisites()
    if msg:
        raise RuntimeError(f"prerequisites: {msg}")
    fea.purge_results()
    fea.write_inp_file()
    fea.ccx_run()
    fea.load_results()
    res = None
    for o in an.Group:
        if o.isDerivedFrom("Fem::FemResultObject"):
            res = o
    if res is None:
        raise RuntimeError(f"CalculiX produced no results; see {workdir}")
    vm = np.array(res.vonMises, float)
    disp = np.array(res.DisplacementLengths, float)
    dvec = np.array([[v.x, v.y, v.z] for v in res.DisplacementVectors], float)
    imax = int(np.argmax(vm))
    node_ids = list(res.NodeNumbers)
    fm = mesh.FemMesh
    p_max = fm.getNodeById(node_ids[imax])
    idmax = int(np.argmax(disp))
    p_dmax = fm.getNodeById(node_ids[idmax])
    out = {
        "solver": "calculix",
        "mesh": {"size_mm": round(mesh_size, 3), "order": int(order), "nodes": n_nodes, "volume_elements": n_vol},
        "von_mises_MPa": {"max": round(float(vm.max()), 3), "p99": round(float(np.percentile(vm, 99)), 3),
                          "p95": round(float(np.percentile(vm, 95)), 3), "mean": round(float(vm.mean()), 3),
                          "max_at": fc_core.r3(p_max, 2)},
        "displacement_mm": {"max": round(float(disp.max()), 5), "max_at": fc_core.r3(p_dmax, 2),
                            "max_vector": dvec[idmax].round(5).tolist()},
        "material": {k: mat[k] for k in ("label", "E", "nu", "yield", "density")},
        "volume_mm3": round(shape.Volume, 2),
        "mass_g": round(shape.Volume * mat["density"] / 1000.0, 2),
        "bcs": used,
        "time_s": {"mesh": round(t_mesh, 2), "total": round(time.time() - t0, 2)},
        "workdir": workdir,
    }
    y = mat["yield"]
    out["safety_factor"] = {"vs_max": round(y / max(vm.max(), 1e-12), 3), "vs_p99": round(y / max(np.percentile(vm, 99), 1e-12), 3)}
    if save_fcstd:
        path = os.path.join(workdir, f"{label}.FCStd")
        doc.saveAs(path)
        out["fcstd"] = path  # open in FreeCAD GUI to view the stress plot
    out["_node_vm"] = vm
    if not keep_doc:
        App.closeDocument(doc.Name)
    return out
