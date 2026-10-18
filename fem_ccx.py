"""Static linear FEM with FreeCAD FEM + gmsh + CalculiX (ccx), headless."""
import os
import re
import time
import uuid

import numpy as np

import fc_core
from fc_core import App, Part, V

import ObjectsFem  # noqa: E402
from femtools import ccxtools  # noqa: E402
from femmesh.gmshtools import GmshTools  # noqa: E402

os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_WARNINGS", "0")


def _orthotropic_inp(path, mat, axis):
    """FreeCAD writes isotropic *ELASTIC; swap in the transversely isotropic printed material."""
    c = fc_core.ccx_engineering_constants(mat, axis)
    with open(path) as f:
        txt = f.read()
    block = ("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n" + ",".join(f"{v:.6g}" for v in c[:8]) + "\n"
             + f"{c[8]:.6g},0\n")
    txt, n = re.subn(r"^\*ELASTIC\s*\n[^\n]*\n", lambda _: block, txt, count=1, flags=re.M | re.I)
    if n != 1:
        raise RuntimeError(f"no *ELASTIC card to replace in {path}")
    with open(path, "w") as f:
        f.write(txt)


def _pstats(v):
    return {"max": round(float(v.max()), 3), "p99": round(float(np.percentile(v, 99)), 3)}


def run_static(shape, material, fixed, loads, mesh_size=None, order=2, workdir=None, label="fem",
               save_fcstd=True, keep_doc=False, matrix_solver=None, build_dir="z", bond_profile=None):
    """fixed: list of {"faces": selector, "dofs": "xyz"} or plain selectors.
    loads: list of {"faces": selector, "force": [fx,fy,fz] N} | {"faces":..., "pressure": MPa}
    (force is total over the selected faces, any direction; pressure acts against the face normal).
    build_dir: print direction for layered (printed) materials -> orthotropic stiffness + layer-adhesion
    check; 'none' = isotropic.
    bond_profile: [[height_mm, factor], ...] along build_dir (height above the bed side of the part): the layer
    bond strength there as a share of yield_z/shear_z (e.g. from print_opt: layers that were cold when the next
    one arrived bond weaker). Interpolated; the layer utilisation is divided by it.
    """
    t0 = time.time()
    mat = fc_core.get_material(material)
    build = fc_core.parse_build_dir(build_dir) if fc_core.is_layered(mat) else None
    # unique per call: parallel analyses of the same model (print_opt orientations) start in the same millisecond
    workdir = workdir or os.path.join(fc_core.RUN_DIR, f"{label}_{int(time.time()*1000)}_{os.getpid()}_{uuid.uuid4().hex[:6]}")
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
        fsel, dofs = fc_core.fixed_entry(fx)
        ids = fc_core.select_faces(shape, fsel)
        refs = [(body, f"Face{i}") for i in ids]
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
        lsel, ld = fc_core.load_entry(ld, k)
        ids = fc_core.select_faces(shape, lsel)
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
    gt = GmshTools(mesh)
    # FreeCAD makes one mesh group per vertex/edge/face and renames them in O(n^2): a mesh-derived
    # solid (20k faces) never finishes. Constraints are matched by geometry, the groups are unused.
    gt.get_group_data = lambda: None
    err = gt.create_mesh()
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
    if build is not None:
        _orthotropic_inp(fea.inp_file_name, mat, build[0])
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
        "material": fc_core.material_report(mat, build),
        "volume_mm3": round(shape.Volume, 2),
        "mass_g": round(shape.Volume * mat["density"] / 1000.0, 2),
        "bcs": used,
        "time_s": {"mesh": round(t_mesh, 2), "total": round(time.time() - t0, 2)},
        "workdir": workdir,
    }
    y = mat["yield"]
    out["safety_factor"] = {"vs_max": round(y / max(vm.max(), 1e-12), 3), "vs_p99": round(y / max(np.percentile(vm, 99), 1e-12), 3)}
    util = vm / y
    if build is not None and "yield_z" in mat:
        import topopt
        s = np.stack([np.array(getattr(res, f"NodeStress{c}"), float) for c in ("XX", "YY", "ZZ", "XY", "YZ", "XZ")], 1)
        sn, tau, lu = topopt.layer_stress(s, build[0], float(mat["yield_z"]), float(mat.get("shear_z", mat["yield_z"])))
        if bond_profile:
            bp = np.asarray(sorted(bond_profile), float)
            ax, sg = build
            xyz = np.array([[(v := fm.getNodeById(n)).x, v.y, v.z] for n in node_ids], float)
            hz = sg * xyz[:, ax]
            hz -= hz.min()
            fac = np.clip(np.interp(hz, bp[:, 0], bp[:, 1]), 0.05, 1.0)
            lu = lu / fac
        il = int(np.argmax(lu))
        out["layer_adhesion"] = {
            "build_dir": out["material"]["build_dir"],
            "sigma_n_MPa": _pstats(sn), "tau_MPa": _pstats(tau),
            "utilization": _pstats(lu), "max_at": fc_core.r3(fm.getNodeById(node_ids[il]), 2),
            "safety_factor": {"vs_max": round(1 / max(lu.max(), 1e-12), 3),
                              "vs_p99": round(1 / max(np.percentile(lu, 99), 1e-12), 3)},
            "criterion": "sqrt((sigma_n/yield_z)^2 + (tau/shear_z)^2), sigma_n = tension across the layers, "
                         "tau = shear on the layer plane",
        }
        if bond_profile:
            out["layer_adhesion"]["bond_profile"] = {"points": len(bond_profile), "min_factor": round(float(bp[:, 1].min()), 3),
                                                     "note": "layer strength scaled by the bond factor at each height"}
        util = np.maximum(util, lu)
        out["_node_layer"] = lu
    gov = {"vs_max": round(1 / max(util.max(), 1e-12), 3), "vs_p99": round(1 / max(np.percentile(util, 99), 1e-12), 3),
           "vs_p95": round(1 / max(np.percentile(util, 95), 1e-12), 3), "vs_mean": round(1 / max(util.mean(), 1e-12), 3)}
    if "layer_adhesion" in out:
        lsf = out["layer_adhesion"]["safety_factor"]
        gov["governed_by"] = "layer_adhesion" if lsf["vs_p99"] < out["safety_factor"]["vs_p99"] else "von_mises"
    out["governing_safety_factor"] = gov
    if save_fcstd:
        path = os.path.join(workdir, f"{label}.FCStd")
        doc.saveAs(path)
        out["fcstd"] = path  # open in FreeCAD GUI to view the stress plot
    out["_node_vm"] = vm
    if not keep_doc:
        App.closeDocument(doc.Name)
    return out
