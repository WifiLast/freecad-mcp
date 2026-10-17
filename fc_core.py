"""FreeCAD helpers shared by the worker: model store, model building, face selectors, materials.

Runs inside the `fc` conda env (FreeCAD 26.x importable after adding $PREFIX/lib to sys.path).
Units everywhere: mm, N, MPa (= N/mm^2), density g/cm^3, mass g.
"""
import ast
import json
import math
import os
import sys

FC_PREFIX = os.environ.get("FREECAD_PREFIX", "/home/wiffzack/miniforge3/envs/fc")
if os.path.join(FC_PREFIX, "lib") not in sys.path:
    sys.path.append(os.path.join(FC_PREFIX, "lib"))

import FreeCAD as App  # noqa: E402
import Part  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.environ.get("FREECAD_MCP_WORK", os.path.join(BASE_DIR, "work"))
MODEL_DIR = os.path.join(WORK_DIR, "models")
RUN_DIR = os.path.join(WORK_DIR, "runs")
EXPORT_DIR = os.path.join(WORK_DIR, "exports")
for _d in (MODEL_DIR, RUN_DIR, EXPORT_DIR):
    os.makedirs(_d, exist_ok=True)

V = App.Vector

# --------------------------------------------------------------------------- materials
# E [MPa], nu, yield [MPa], density [g/cm^3]
MATERIALS = {
    "steel":      {"E": 210000, "nu": 0.30, "yield": 235, "density": 7.85, "label": "Steel S235"},
    "steel_s355": {"E": 210000, "nu": 0.30, "yield": 355, "density": 7.85, "label": "Steel S355"},
    "stainless":  {"E": 193000, "nu": 0.29, "yield": 215, "density": 8.00, "label": "Stainless 1.4301"},
    "aluminium":  {"E": 69000,  "nu": 0.33, "yield": 240, "density": 2.70, "label": "Aluminium 6061-T6"},
    "titanium":   {"E": 114000, "nu": 0.34, "yield": 880, "density": 4.43, "label": "Ti-6Al-4V"},
    "pla":        {"E": 3500,   "nu": 0.36, "yield": 50,  "density": 1.24, "label": "PLA (printed, ~100% infill)"},
    "petg":       {"E": 2100,   "nu": 0.38, "yield": 45,  "density": 1.27, "label": "PETG (printed)"},
    "abs":        {"E": 2200,   "nu": 0.35, "yield": 38,  "density": 1.04, "label": "ABS (printed)"},
    "nylon":      {"E": 1700,   "nu": 0.40, "yield": 45,  "density": 1.01, "label": "PA12 (SLS)"},
}
MATERIALS["aluminum"] = MATERIALS["aluminium"]


def get_material(m):
    if m is None:
        m = "steel"
    if isinstance(m, str):
        key = m.lower().strip()
        if key not in MATERIALS:
            raise ValueError(f"unknown material '{m}', known: {', '.join(sorted(MATERIALS))} or a dict {{E,nu,yield,density}}")
        return dict(MATERIALS[key], key=key)
    base = dict(MATERIALS.get(str(m.get("base", "steel")).lower(), MATERIALS["steel"]))
    base.update({k: m[k] for k in ("E", "nu", "yield", "density", "label") if k in m})
    base.setdefault("label", "custom")
    base["key"] = "custom"
    return base


# --------------------------------------------------------------------------- safe expressions
_ALLOWED_FUNCS = {"min": min, "max": max, "abs": abs, "sqrt": math.sqrt, "sin": math.sin,
                  "cos": math.cos, "tan": math.tan, "radians": math.radians, "pi": math.pi,
                  "round": round}


def eval_expr(expr, params):
    """Evaluate a numeric expression string like 't*2+1' with params; whitelisted AST only."""
    if not isinstance(expr, str):
        return expr
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
                             ast.Call, ast.operator, ast.unaryop, ast.Tuple, ast.List)):
            if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                raise ValueError(f"function not allowed in '{expr}'")
            continue
        raise ValueError(f"expression element {type(node).__name__} not allowed in '{expr}'")
    env = dict(_ALLOWED_FUNCS)
    env.update(params or {})
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env)


def resolve_params(obj, params):
    """Recursively replace strings that look like expressions with numbers (keys untouched)."""
    if isinstance(obj, dict):
        return {k: (v if k in ("op", "plane", "axis_name", "name", "type", "side", "mode") and isinstance(v, str)
                    else resolve_params(v, params)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_params(v, params) for v in obj]
    if isinstance(obj, str):
        try:
            return eval_expr(obj, params)
        except Exception:
            return obj
    return obj


# --------------------------------------------------------------------------- model building
_AXES = {"x": V(1, 0, 0), "y": V(0, 1, 0), "z": V(0, 0, 1),
         "-x": V(-1, 0, 0), "-y": V(0, -1, 0), "-z": V(0, 0, -1)}


def _vec(v, default=(0, 0, 0)):
    if v is None:
        v = default
    if isinstance(v, str):
        return V(_AXES[v.lower()])
    return V(*[float(c) for c in v])


def _plane_xyz(plane, a, b, c=0.0):
    """Map 2D sketch coords (a,b) + offset c to 3D for plane xy / xz / yz."""
    plane = (plane or "xy").lower()
    if plane == "xy":
        return V(a, b, c)
    if plane == "xz":
        return V(a, c, b)
    if plane == "yz":
        return V(c, a, b)
    raise ValueError("plane must be xy, xz or yz")


_PLANE_NORMAL = {"xy": V(0, 0, 1), "xz": V(0, 1, 0), "yz": V(1, 0, 0)}


def _polygon_face(points, plane, offset):
    pts = [_plane_xyz(plane, float(p[0]), float(p[1]), offset) for p in points]
    if (pts[0] - pts[-1]).Length > 1e-9:
        pts.append(pts[0])
    return Part.Face(Part.makePolygon(pts))


def build_primitive(op):
    """One op dict -> Part shape. Kinds: box, cylinder, sphere, cone, torus, extrude, revolve, plate."""
    at = _vec(op.get("at"))
    if "box" in op:
        sx, sy, sz = [float(c) for c in op["box"]]
        if op.get("center"):
            at = at - V(sx / 2, sy / 2, sz / 2)
        return Part.makeBox(sx, sy, sz, at)
    if "cylinder" in op:
        c = op["cylinder"]
        r, h = float(c["r"] if "r" in c else c["d"] / 2), float(c["h"])
        ax = _vec(op.get("axis", "z"))
        if op.get("center"):
            at = at - ax.normalize() * (h / 2) if ax.Length else at
        return Part.makeCylinder(r, h, at, ax)
    if "sphere" in op:
        return Part.makeSphere(float(op["sphere"]), at)
    if "cone" in op:
        c = op["cone"]
        return Part.makeCone(float(c["r1"]), float(c["r2"]), float(c["h"]), at, _vec(op.get("axis", "z")))
    if "torus" in op:
        c = op["torus"]
        return Part.makeTorus(float(c["R"]), float(c["r"]), at, _vec(op.get("axis", "z")))
    if "extrude" in op:
        e = op["extrude"]
        plane = e.get("plane", "xy")
        face = _polygon_face(e["points"], plane, float(e.get("offset", 0)))
        for hole in e.get("holes", []):  # circular holes [a, b, r]
            a, b, r = [float(x) for x in hole]
            circ = Part.Face(Part.Wire(Part.makeCircle(r, _plane_xyz(plane, a, b, float(e.get("offset", 0))), _PLANE_NORMAL[plane])))
            face = face.cut(circ)
        solid = face.extrude(_PLANE_NORMAL[plane] * float(e["h"]))
        solid.translate(at)
        return solid
    if "revolve" in op:
        e = op["revolve"]  # profile points (r, z) in xz plane, revolved around z
        face = _polygon_face(e["points"], "xz", 0.0)
        solid = face.revolve(V(0, 0, 0), V(0, 0, 1), float(e.get("angle", 360)))
        solid.translate(at)
        return solid
    raise ValueError(f"unknown op {sorted(op)}; use box/cylinder/sphere/cone/torus/extrude/revolve")


def build_from_ops(ops, params=None):
    ops = resolve_params(ops, params or {})
    shape = None
    for i, op in enumerate(ops):
        if "fillet" in op or "chamfer" in op:
            if shape is None:
                raise ValueError("fillet/chamfer needs a shape first")
            r = float(op.get("fillet", op.get("chamfer")))
            edges = shape.Edges
            if "edges" in op and op["edges"] != "all":
                edges = [shape.Edges[int(str(e).replace("Edge", "")) - 1] for e in op["edges"]]
            elif "edges_near" in op:
                edges = select_edges_near(shape, op["edges_near"], float(op.get("tol", 0.5)))
            try:
                shape = shape.makeFillet(r, edges) if "fillet" in op else shape.makeChamfer(r, edges)
            except Exception as e:
                raise ValueError(f"op {i}: fillet/chamfer failed ({e}); try a smaller radius or fewer edges")
            continue
        prim = build_primitive(op)
        if op.get("rotate"):  # [axis_vec, angle_deg] around 'at' or [ax,ay,az,deg]
            rot = op["rotate"]
            prim.rotate(_vec(op.get("pivot", op.get("at"))), _vec(rot[:3]), float(rot[3]))
        for m in op.get("mirror", []) or []:  # list of axis names: mirror copy across plane through origin
            n = {"x": V(1, 0, 0), "y": V(0, 1, 0), "z": V(0, 0, 1)}[m]
            prim = prim.fuse(prim.mirror(_vec(op.get("mirror_at")), n))
        for off in op.get("copies", []) or []:
            prim = prim.fuse(_moved(build_primitive(op), op, off))
        mode = "cut" if op.get("cut") else "common" if op.get("common") else "fuse"
        if shape is None:
            if mode != "fuse":
                raise ValueError(f"op {i}: first op cannot be {mode}")
            shape = prim
        elif mode == "cut":
            shape = shape.cut(prim)
        elif mode == "common":
            shape = shape.common(prim)
        else:
            shape = shape.fuse(prim)
    if shape is None:
        raise ValueError("no ops given")
    return clean_shape(shape)


def _moved(prim, op, off):
    if op.get("rotate"):
        rot = op["rotate"]
        prim.rotate(_vec(op.get("pivot", op.get("at"))), _vec(rot[:3]), float(rot[3]))
    prim.translate(_vec(off))
    return prim


def clean_shape(shape):
    try:
        shape = shape.removeSplitter()
    except Exception:
        pass
    solids = shape.Solids
    if len(solids) == 1:
        return solids[0]
    if len(solids) > 1:
        return shape
    return shape


def run_script(code, params=None):
    """FreeCAD python code that must assign `shape` (Part.Shape). params injected as variables."""
    env = {"App": App, "FreeCAD": App, "Part": Part, "V": V, "Vector": V, "math": math, "params": params or {}}
    env.update(params or {})
    exec(compile(code, "<model_script>", "exec"), env)
    shape = env.get("shape")
    if shape is None or not hasattr(shape, "Volume"):
        raise ValueError("script must assign a Part shape to the variable `shape`")
    return clean_shape(shape)


def import_file(path):
    path = os.path.expanduser(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".step", ".stp", ".iges", ".igs", ".brep", ".brp"):
        s = Part.Shape()
        s.read(path)
        return clean_shape(s)
    if ext == ".stl" or ext == ".obj":
        import Mesh
        m = Mesh.Mesh(path)
        return mesh_to_solid(m)
    if ext == ".fcstd":
        doc = App.openDocument(path)
        shapes = [o.Shape for o in doc.Objects if hasattr(o, "Shape") and o.Shape.Solids and getattr(o, "Visibility", True)]
        # take the visible tip solids (Body / features); fuse if several
        if not shapes:
            raise ValueError("no solids in document")
        best = max(shapes, key=lambda s: s.Volume)
        App.closeDocument(doc.Name)
        return clean_shape(best)
    raise ValueError(f"unsupported file type {ext}")


def _shell_solid(mesh, tol):
    sh = Part.Shape()
    sh.makeShapeFromMesh(mesh.Topology, tol)
    try:
        solid = Part.Solid(Part.Shell(sh.Faces))
    except Exception:
        solid = Part.makeSolid(sh)
    if solid.Volume < 0:
        solid.reverse()
    return solid


def mesh_to_solid(mesh, tol=None):
    """Mesh -> Part solid, one shell per connected component (a shell holding several bodies, or
    sewing nearly touching bodies together, gives an invalid solid). tol: sewing tolerance (mm)."""
    if tol is None:
        tol = max(mesh.BoundBox.DiagonalLength * 1e-7, 1e-6)
    try:
        comps = mesh.getSeparateComponents()
    except Exception:
        comps = [mesh]
    if len(comps) <= 1:
        return _shell_solid(mesh, tol)
    solids = sorted((_shell_solid(c, tol) for c in comps), key=lambda x: -abs(x.Volume))
    solids = [x for x in solids if abs(x.Volume) > 1e-9]
    # a component nested in an odd number of larger ones is a cavity (e.g. a hollowed interior):
    # cut it, never fuse it - fusing would fill the void again
    bodies, voids = [], []
    for i, x in enumerate(solids):
        p = x.Vertexes[0].Point
        depth = sum(1 for y in solids[:i] if y.BoundBox.isInside(p) and y.isInside(p, tol, False))
        (voids if depth % 2 else bodies).append(x)
    try:
        res = bodies[0].fuse(bodies[1:]) if len(bodies) > 1 else bodies[0]
        if voids:
            res = res.cut(voids)
        res = res.Solids[0] if len(res.Solids) == 1 else res
        if res.isValid():
            return res
    except Exception:
        pass
    return Part.makeCompound(bodies)


# --------------------------------------------------------------------------- store
def model_paths(name):
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_.")
    if not safe:
        raise ValueError("bad model name")
    return os.path.join(MODEL_DIR, safe + ".brep"), os.path.join(MODEL_DIR, safe + ".json")


def save_model(name, shape, meta):
    brep, js = model_paths(name)
    shape.exportBrep(brep)
    meta = dict(meta)
    meta["name"] = name
    with open(js, "w") as f:
        json.dump(meta, f, indent=1)
    return brep


def load_model(name):
    brep, js = model_paths(name)
    if not os.path.exists(brep):
        raise ValueError(f"model '{name}' not found (list_models)")
    s = Part.Shape()
    s.read(brep)
    meta = json.load(open(js)) if os.path.exists(js) else {"name": name}
    if s.ShapeType == "Compound" and len(s.Solids) == 1:
        s = s.Solids[0]
    return s, meta


def list_models():
    out = []
    for fn in sorted(os.listdir(MODEL_DIR)):
        if fn.endswith(".json"):
            try:
                m = json.load(open(os.path.join(MODEL_DIR, fn)))
                out.append({k: m.get(k) for k in ("name", "source", "volume_mm3", "bbox", "derived_from", "faces") if k in m})
            except Exception:
                pass
    return out


# --------------------------------------------------------------------------- inspection
def r3(v, nd=3):
    return [round(float(c), nd) for c in (v.x, v.y, v.z)] if hasattr(v, "x") else [round(float(c), nd) for c in v]


def face_info(face, idx):
    s = face.Surface
    typ = type(s).__name__.replace("Geom", "")
    info = {"id": idx, "type": typ, "area": round(face.Area, 2), "center": r3(face.CenterOfMass, 2)}
    try:
        u0, u1, v0, v1 = face.ParameterRange
        n = face.normalAt((u0 + u1) / 2, (v0 + v1) / 2)
        if typ == "Plane":
            info["normal"] = r3(n, 3)
    except Exception:
        pass
    if typ in ("Cylinder", "Cone", "Sphere"):
        try:
            info["radius"] = round(s.Radius, 3)
            if hasattr(s, "Axis"):
                info["axis"] = r3(s.Axis, 3)
        except Exception:
            pass
    return info


def bbox_list(shape):
    b = shape.BoundBox
    return [round(b.XMin, 3), round(b.YMin, 3), round(b.ZMin, 3), round(b.XMax, 3), round(b.YMax, 3), round(b.ZMax, 3)]


def summary(shape, name=None, material=None, faces=True, max_faces=60):
    out = {}
    if name:
        out["name"] = name
    out["volume_mm3"] = round(shape.Volume, 2)
    out["bbox"] = bbox_list(shape)
    out["n_faces"] = len(shape.Faces)
    out["valid"] = bool(shape.isValid())
    out["solids"] = len(shape.Solids)
    if material:
        mat = get_material(material)
        out["mass_g"] = round(shape.Volume * mat["density"] / 1000.0, 2)
    if faces:
        if len(shape.Faces) <= max_faces:
            out["faces"] = [face_info(f, i + 1) for i, f in enumerate(shape.Faces)]
        else:
            out["faces_note"] = (f"{len(shape.Faces)} faces (mesh-derived?) - listing skipped; select with "
                                 "{side|normal|near|box|on_model} selectors")
    return out


# --------------------------------------------------------------------------- selectors
def _axis_of(side):
    side = side.lower()
    ax = {"x": 0, "y": 1, "z": 2}[side[0]]
    return ax, side.endswith("max")


def _face_normal(f):
    u0, u1, v0, v1 = f.ParameterRange
    return f.normalAt((u0 + u1) / 2, (v0 + v1) / 2)


def select_faces(shape, sel, tol=None):
    """Resolve a selector into 1-based face ids.

    Accepted: 3 | "Face3" | [3, 5] | {"faces":[..]} | {"side":"xmin"} | {"normal":[0,0,1] or "+z", "near":[..]?}
    | {"near":[x,y,z]} (single closest face) | {"box":[[x0,y0,z0],[x1,y1,z1]]} (face centers inside)
    | {"type":"cylinder","radius":r?,"near":..?} | {"on_model":name,"faces":<selector>,"tol":mm}
    | {"plane":"z","value":10} (planar faces lying on z=10).  Combine with "and": [sel, sel].
    """
    faces = shape.Faces
    bb = shape.BoundBox
    diag = bb.DiagonalLength
    tol = tol if tol is not None else max(1e-3 * diag, 1e-4)

    if isinstance(sel, (int, float)) or (isinstance(sel, str) and sel.lower().startswith("face")):
        idx = int(str(sel).lower().replace("face", ""))
        if not 1 <= idx <= len(faces):
            raise ValueError(f"face {idx} out of range 1..{len(faces)}")
        return [idx]
    if isinstance(sel, str):
        if sel.lower().replace("-", "").replace("+", "") in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            return select_faces(shape, {"side": sel}, tol)
        raise ValueError(f"bad selector '{sel}'")
    if isinstance(sel, list):
        out = []
        for s in sel:
            for i in select_faces(shape, s, tol):
                if i not in out:
                    out.append(i)
        return out
    if not isinstance(sel, dict):
        raise ValueError(f"bad selector {sel!r}")
    if "and" in sel:
        sets = [set(select_faces(shape, s, tol)) for s in sel["and"]]
        return sorted(set.intersection(*sets))
    if "faces" in sel and "on_model" not in sel:
        return select_faces(shape, sel["faces"], tol)
    if "on_model" in sel:
        other, _ = load_model(sel["on_model"])
        ref_ids = select_faces(other, sel.get("faces"), None)
        other_faces = other.Faces
        return faces_on(shape, [other_faces[i - 1] for i in ref_ids], float(sel.get("tol", 0.5)))

    cand = list(range(1, len(faces) + 1))
    if "side" in sel:
        ax, is_max = _axis_of(sel["side"])
        ext = [bb.XMax, bb.YMax, bb.ZMax][ax] if is_max else [bb.XMin, bb.YMin, bb.ZMin][ax]
        stol = float(sel.get("tol", tol))
        keep = []
        for i in cand:
            fb = faces[i - 1].BoundBox
            lo, hi = [fb.XMin, fb.YMin, fb.ZMin][ax], [fb.XMax, fb.YMax, fb.ZMax][ax]
            if abs(lo - ext) <= stol and abs(hi - ext) <= stol:
                keep.append(i)
        cand = keep
    if "plane" in sel:
        ax = {"x": 0, "y": 1, "z": 2}[sel["plane"].lower()]
        val = float(sel["value"])
        ptol = float(sel.get("tol", tol))
        cand = [i for i in cand
                if abs([faces[i - 1].BoundBox.XMin, faces[i - 1].BoundBox.YMin, faces[i - 1].BoundBox.ZMin][ax] - val) <= ptol
                and abs([faces[i - 1].BoundBox.XMax, faces[i - 1].BoundBox.YMax, faces[i - 1].BoundBox.ZMax][ax] - val) <= ptol]
    if "normal" in sel:
        n = _vec(sel["normal"].replace("+", "") if isinstance(sel["normal"], str) else sel["normal"]).normalize()
        cosmin = math.cos(math.radians(float(sel.get("tol_deg", 5))))
        keep = []
        for i in cand:
            f = faces[i - 1]
            if type(f.Surface).__name__ != "Plane" and "Plane" not in type(f.Surface).__name__:
                continue
            try:
                if _face_normal(f).dot(n) >= cosmin:
                    keep.append(i)
            except Exception:
                pass
        cand = keep
    if "type" in sel:
        t = sel["type"].lower()
        cand = [i for i in cand if t in type(faces[i - 1].Surface).__name__.lower()]
        if "radius" in sel:
            r = float(sel["radius"])
            cand = [i for i in cand if hasattr(faces[i - 1].Surface, "Radius") and abs(faces[i - 1].Surface.Radius - r) <= max(tol, 1e-3 * r)]
    if "box" in sel:
        (x0, y0, z0), (x1, y1, z1) = sel["box"]
        keep = []
        for i in cand:
            c = faces[i - 1].CenterOfMass
            if x0 - tol <= c.x <= x1 + tol and y0 - tol <= c.y <= y1 + tol and z0 - tol <= c.z <= z1 + tol:
                keep.append(i)
        cand = keep
    if "near" in sel:
        p = _vec(sel["near"])
        vtx = Part.Vertex(p)
        dists = [(faces[i - 1].distToShape(vtx)[0], i) for i in cand]
        if not dists:
            return []
        dmin = min(d for d, _ in dists)
        n_pick = int(sel.get("count", 1))
        dists.sort()
        cand = [i for d, i in dists[:n_pick] if d <= dmin + tol or n_pick > 1]
    if not cand:
        raise ValueError(f"selector {json.dumps(sel)} matched no faces")
    return cand


def select_edges_near(shape, points, tol):
    out = []
    for p in points:
        vtx = Part.Vertex(_vec(p))
        best = min(shape.Edges, key=lambda e: e.distToShape(vtx)[0])
        if best.distToShape(vtx)[0] <= tol and best not in out:
            out.append(best)
    return out


def face_samples(face, spacing):
    """Area-weighted sample points on a face: (points Nx3, weights N) from a fine tessellation."""
    import numpy as np
    import MeshPart
    m = MeshPart.meshFromShape(Shape=face, MaxLength=max(spacing, 1e-3))
    pts, w = [], []
    for fac in m.Facets:
        a, b, c = [V(*p) for p in fac.Points]
        pts.append(r3((a + b + c) * (1 / 3.0), 9))
        w.append(fac.Area)
    if not pts:  # degenerate tessellation, fall back to center
        pts, w = [r3(face.CenterOfMass, 9)], [face.Area]
    return np.array(pts, float), np.array(w, float)


def faces_on(shape, ref_faces, tol):
    """Faces of `shape` whose sampled points all lie within tol of the reference faces (KD-tree)."""
    import numpy as np
    from scipy.spatial import cKDTree
    ref = np.vstack([face_samples(f, max(tol, 0.5))[0] for f in ref_faces] +
                    [np.array([r3(v.Point, 9) for f in ref_faces for v in f.Vertexes])])
    tree = cKDTree(ref)
    spacing = max(tol, 0.5)
    out = []
    for i, f in enumerate(shape.Faces):
        verts = np.array([r3(v.Point, 9) for v in f.Vertexes] + [r3(f.CenterOfMass, 9)])
        d, _ = tree.query(verts)
        if np.all(d <= tol + spacing):
            out.append(i + 1)
    if not out:
        raise ValueError("on_model: no faces of this shape lie on the reference faces; raise tol")
    return out
