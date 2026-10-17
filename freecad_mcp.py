"""FreeCAD MCP: build/import parts, simulate forces (FreeCAD FEM + CalculiX) and reduce material
(voxel SIMP topology optimisation, or mass-minimal dimensions under a stress limit).

Runs in the `fc` conda env (FreeCAD 26.x, gmsh, CalculiX, scikit-sparse). Every job runs in its own
worker process (fc_worker.py) so a FreeCAD/gmsh crash cannot take the server down.
Env: FREECAD_MCP_HOST/PORT/PATH (0.0.0.0:2022/freecad), FREECAD_MCP_TIMEOUT (s), FREECAD_MCP_WORK.
Units: mm, N, MPa, g.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

from fastmcp import Context, FastMCP
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("FREECAD_MCP_PYTHON", sys.executable)
WORKER = os.path.join(BASE_DIR, "fc_worker.py")
TIMEOUT = float(os.environ.get("FREECAD_MCP_TIMEOUT", "3600"))
SERVER_HOST = os.environ.get("FREECAD_MCP_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("FREECAD_MCP_PORT", "2022"))
SERVER_PATH = os.environ.get("FREECAD_MCP_PATH", "/freecad")
TMP_DIR = os.path.join(os.environ.get("FREECAD_MCP_WORK", os.path.join(BASE_DIR, "work")), "jobs")
os.makedirs(TMP_DIR, exist_ok=True)
PREVIEW_MAX = 1100
GPU_PYTHON = os.environ.get("FREECAD_GPU_PYTHON", "/mnt/data/miniconda3/envs/py310_amd/bin/python")
GPU_SOCKET = os.environ.get("FREECAD_GPU_SOCKET", os.path.join(BASE_DIR, "work", "gpu.sock"))
GPU_IDLE_S = os.environ.get("FREECAD_GPU_IDLE", "600")  # shared helper frees its VRAM after this
_gpu_proc: Optional[asyncio.subprocess.Process] = None

INSTRUCTIONS = """FreeCAD part design + force simulation + material reduction. Units: mm, N, MPa, g.
Workflow: create_model (ops/script/file) -> read the face list -> analyze (fixed + loads) ->
(optimize_mesh repairs/decimates meshes on the GPU; export_web writes gltfpack-optimised .glb) ->
optimize_topology (remove material where it carries no load) or optimize_dimensions (parametric
model, minimal mass with stress <= yield/safety_factor) -> export_model (STEP/STL/FCStd).
Face selectors (fixed/loads/keep): 3 | "Face3" | [3,5] | "xmin".."zmax" | {"side":"zmin"} |
{"normal":"+z" or [0,0,1]} | {"near":[x,y,z]} | {"type":"cylinder","radius":4} | {"box":[[x0,y0,z0],[x1,y1,z1]]}
| {"plane":"z","value":10} | {"and":[sel,sel]} | {"on_model":"other","faces":sel}.
Loads: {"faces":sel,"force":[fx,fy,fz]} (total N over the faces) or {"faces":sel,"pressure":MPa}.
Fixed: sel or {"faces":sel,"dofs":"z"} (only z blocked). Materials: steel, steel_s355, stainless,
aluminium, titanium, pla, petg, abs, nylon or {"E":MPa,"nu":..,"yield":MPa,"density":g/cm3}."""

mcp = FastMCP("freecad", instructions=INSTRUCTIONS)


async def _ensure_gpu_helper() -> None:
    """Start the shared torch/GPU helper (warm ROCm context for all jobs) unless it is running."""
    global _gpu_proc
    if os.environ.get("FREECAD_MCP_DEVICE", "auto").lower() == "cpu" or os.environ.get("FREECAD_GPU_SHARED", "1") == "0":
        return
    if _gpu_proc is not None and _gpu_proc.returncode is None:
        # alive (maybe still initialising torch): never start a second one, it would steal the socket
        for _ in range(100):
            if os.path.exists(GPU_SOCKET) or _gpu_proc.returncode is not None:
                break
            await asyncio.sleep(0.2)
        if _gpu_proc.returncode is None:
            return
    if os.path.exists(GPU_SOCKET):  # helper left by an earlier server instance still answering?
        try:
            _, w = await asyncio.wait_for(asyncio.open_unix_connection(GPU_SOCKET), 2.0)
            w.close()
            return
        except Exception:
            pass
    if not os.path.exists(GPU_PYTHON):
        return
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}
    env["PYTHONNOUSERSITE"] = "1"
    _gpu_proc = await asyncio.create_subprocess_exec(
        GPU_PYTHON, os.path.join(BASE_DIR, "gpu_solver.py"), "--socket", GPU_SOCKET, "--idle", GPU_IDLE_S,
        cwd=BASE_DIR, env=env, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    for _ in range(100):  # torch + ROCm init takes a few seconds
        if os.path.exists(GPU_SOCKET) or _gpu_proc.returncode is not None:
            break
        await asyncio.sleep(0.2)


async def _run(action: str, args: Dict[str, Any], ctx: Optional[Context] = None, timeout: float = TIMEOUT) -> Dict[str, Any]:
    job = {k: v for k, v in args.items() if v is not None}
    job["action"] = action
    fd, jpath = tempfile.mkstemp(prefix=f"{action}_", suffix=".json", dir=TMP_DIR)
    with os.fdopen(fd, "w") as f:
        json.dump(job, f)
    opath = jpath + ".out"
    if action in ("optimize_topology", "optimize_mesh", "create_model", "export_web") and (args.get("device") or "auto") != "cpu" \
            and (action != "create_model" or str(args.get("file") or "").lower().endswith((".stl", ".obj", ".ply", ".off"))):
        await _ensure_gpu_helper()
    env = dict(os.environ, KMP_AFFINITY="disabled", KMP_WARNINGS="0", PYTHONNOUSERSITE="1",
               FREECAD_GPU_SOCKET=GPU_SOCKET)
    proc = await asyncio.create_subprocess_exec(PYTHON, WORKER, jpath, opath, cwd=BASE_DIR, env=env,
                                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    tail: List[str] = []
    t0 = time.time()

    async def pump():
        n = 0
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            s = line.decode(errors="replace").rstrip()
            if not s or s.startswith("OMP:"):
                continue
            tail.append(s)
            del tail[:-40]
            if ctx is not None and s.startswith("["):
                n += 1
                try:
                    await ctx.report_progress(progress=n, total=None, message=s[:200])
                except Exception:
                    pass

    try:
        await asyncio.wait_for(asyncio.gather(pump(), proc.wait()), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"{action} timed out after {timeout:.0f}s; lower resolution/mesh size. Last log: {' | '.join(tail[-3:])}")
    finally:
        try:
            os.remove(jpath)
        except OSError:
            pass
    if not os.path.exists(opath):
        raise RuntimeError(f"worker crashed (exit {proc.returncode}) after {time.time() - t0:.0f}s: " + " | ".join(tail[-6:]))
    with open(opath) as f:
        res = json.load(f)
    os.remove(opath)
    if not res.get("ok"):
        raise RuntimeError(res.get("error", "unknown error"))
    return res["result"]


def _preview(path: str):
    from PIL import Image as PILImage
    img = PILImage.open(path)
    img.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=82)
    return Image(data=buf.getvalue(), format="jpeg").to_image_content()


def _with_previews(result: Dict[str, Any], show: bool) -> ToolResult:
    paths = []
    if show:
        if result.get("preview"):
            paths.append(result["preview"])
        for v in result.get("verify_calculix") or []:
            if isinstance(v, dict) and v.get("preview"):
                paths.append(v["preview"])
    content: List[Any] = [TextContent(type="text", text=json.dumps(result, indent=1))]
    for p in paths:
        try:
            content.append(_preview(p))
        except Exception:
            pass
    return ToolResult(content=content, structured_content=result)


Selector = Union[int, str, List[Any], Dict[str, Any]]


@mcp.tool
async def list_models() -> Dict[str, Any]:
    """List stored models (name, volume, bbox, origin) and the material presets."""
    return await _run("list_models", {})


@mcp.tool
async def create_model(name: str, ops: List[Dict[str, Any]] = None, script: str = None, file: str = None,
                       params: Dict[str, float] = None, material: Union[str, Dict[str, Any]] = None,
                       list_faces: bool = True, mesh_max_faces: int = 20000) -> Dict[str, Any]:
    """Create or import a solid and store it as `name`. Returns volume, mass, bbox and the face list
    (id, type, area, center, normal) used to pick supports and loads.
    Give ONE of:
    - ops: CSG list, applied in order (fuse by default; "cut":true subtracts, "common":true intersects):
      {"box":[sx,sy,sz],"at":[x,y,z],"center":false} | {"cylinder":{"r":5,"h":20},"at":[..],"axis":"z"|[..]}
      | {"sphere":r,"at":..} | {"cone":{"r1","r2","h"}} | {"torus":{"R","r"}}
      | {"extrude":{"points":[[a,b],..],"h":10,"plane":"xy|xz|yz","offset":0,"holes":[[a,b,r]]}}
      | {"revolve":{"points":[[r,z],..],"angle":360}} | {"fillet":r,"edges":"all"|[ids]|"edges_near":[[x,y,z]]}
      | {"chamfer":r,...}. Extras per op: "rotate":[ax,ay,az,deg], "pivot":[..], "mirror":["x"], "copies":[[dx,dy,dz],..].
      Any number may be an expression string using params, e.g. {"box":["L","w","t*2"]}.
    - script: FreeCAD Python that assigns `shape` (Part, App, V available; params as variables).
    - file: path to STEP/IGES/BREP/FCStd or a mesh (STL/OBJ/PLY/OFF). Meshes are repaired and, above
      mesh_max_faces (default 20000), decimated on the GPU (CuMesh) before conversion to a solid.
    """
    return await _run("create_model", dict(name=name, ops=ops, script=script, file=file, params=params,
                                           material=material, list_faces=list_faces,
                                           mesh_max_faces=mesh_max_faces))


@mcp.tool
async def inspect_model(name: str, material: Union[str, Dict[str, Any]] = None, select: Selector = None,
                        list_faces: bool = True, max_faces: int = 60) -> Dict[str, Any]:
    """Volume/mass/bbox/faces of a stored model. `select` = a face selector to test which face ids it picks."""
    return await _run("inspect_model", dict(name=name, material=material, select=select, list_faces=list_faces,
                                            max_faces=max_faces))


@mcp.tool
async def analyze(name: str, fixed: List[Selector], loads: List[Dict[str, Any]],
                  material: Union[str, Dict[str, Any]] = None, mesh_size: float = None, order: int = 2,
                  safety_factor: float = None, stress_metric: str = "p99", show: bool = True,
                  ctx: Context = None) -> ToolResult:
    """Static stress simulation (FreeCAD FEM, gmsh tetra mesh, CalculiX). fixed: list of face selectors
    or {"faces":sel,"dofs":"xyz"|"z"|..}. loads: [{"faces":sel,"force":[fx,fy,fz]}] (total N) or
    [{"faces":sel,"pressure":MPa}]. Returns von Mises max/p99/p95 (MPa), max displacement (mm),
    safety factor vs yield, mass, a stress preview and an .FCStd with results (open in FreeCAD GUI).
    The max at sharp support edges is a mesh-dependent singularity; p99 is the robust value.
    mesh_size in mm (auto ~3 elements over the thinnest dimension)."""
    r = await _run("analyze", dict(name=name, fixed=fixed, loads=loads, material=material, mesh_size=mesh_size,
                                   order=order, safety_factor=safety_factor, stress_metric=stress_metric,
                                   render=show), ctx)
    return _with_previews(r, show)


@mcp.tool
async def optimize_topology(name: str, fixed: List[Selector], loads: List[Dict[str, Any]] = None,
                            load_cases: List[List[Dict[str, Any]]] = None,
                            material: Union[str, Dict[str, Any]] = None,
                            volume_fraction: float = None, target_safety_factor: float = None,
                            stress_metric: str = "p99", resolution: int = 50,
                            keep: List[Any] = None, keep_thickness: float = None, shell: float = None, symmetry: List[str] = None,
                            filter_radius: float = None, iterations: int = None, quality: str = "draft", minify: bool = True, minify_error: float = None, out_name: str = None,
                            verify: bool = True, verify_mesh_size: float = None, smooth: int = 8,
                            device: str = None, target_faces: int = 6000, show: bool = True,
                            ctx: Context = None) -> ToolResult:
    """Remove material that carries no load (SIMP topology optimisation on a voxel grid inside the
    part's own volume; the result never grows outside the original).
    - Same fixed/loads as analyze; load_cases = list of load lists optimised together (stiff for all).
    - Either volume_fraction (keep e.g. 0.3 = 30 % of the material, default 0.35) or
      target_safety_factor (searches the smallest volume whose voxel stress <= yield/SF; ~4-6 runs).
    - resolution: voxels along the longest side (30 fast .. 100 detailed; GPU makes 60-100 practical).
      device: auto (GPU solver for >4k voxels, CPU fallback) | gpu | cpu. filter_radius (mm) = min
      member thickness scale (default 1.5 voxels). symmetry: ["y"] etc. mirror planes through the bbox centre.
    - keep: faces kept solid ({"faces":sel,"thickness":mm}) or {"region":[[x0,y0,z0],[x1,y1,z1]]};
      support/load faces are always kept with keep_thickness (default 1 voxel).
    - shell (mm): 3D-print mode - a solid skin of this wall thickness is kept on every surface (outside
      and holes), so the part looks unchanged and only the interior is hollowed into load paths.
      volume_fraction / the SF search then refer to the interior share kept (default 0.25, like infill %).
      For a mesh source (STL/OBJ/PLY/OFF) the exported STL is the untouched original mesh plus the
      cavity shells, cut on a fine grid with an exact skin - so resolution only sets how detailed the
      inner load paths are (80-120 is plenty); for CAD sources use voxel <= shell/2.
    - minify (default true): final error-bounded simplification of the exported STL/GLB; minify_error
      in mm (default 0.05 -> ~65 % fewer triangles, mean deviation ~0.02 mm). minify=false keeps all.
    - quality: draft (default, 25 iterations, loose solves - fast, near-optimal) | normal | fine;
      iterations overrides it.
    - Result stored as out_name (default <name>_opt): smoothed iso-surface trimmed with the original,
      so mounting faces/holes stay exact; STL+STEP exported. verify=True re-runs CalculiX on the
      optimised and original parts with the same supports/loads (adds 0.5-3 min).
      The iso-surface is repaired and decimated to target_faces on the GPU (CuMesh).
    Typical run: 1-4 min."""
    r = await _run("optimize_topology", dict(
        name=name, fixed=fixed, loads=loads, load_cases=load_cases, material=material,
        volume_fraction=volume_fraction, target_safety_factor=target_safety_factor, stress_metric=stress_metric,
        resolution=resolution, keep=keep, keep_thickness=keep_thickness, shell=shell, symmetry=symmetry,
        filter_radius=filter_radius, iterations=iterations, quality=quality, minify=minify, minify_error=minify_error, out_name=out_name, verify=verify,
        verify_mesh_size=verify_mesh_size, smooth=smooth, device=device, target_faces=target_faces), ctx)
    return _with_previews(r, show)


@mcp.tool
async def optimize_dimensions(bounds: Dict[str, Any], fixed: List[Selector], loads: List[Dict[str, Any]],
                              ops: List[Dict[str, Any]] = None, script: str = None,
                              params: Dict[str, float] = None, material: Union[str, Dict[str, Any]] = None,
                              safety_factor: float = 2.0, stress_metric: str = "p99",
                              max_displacement: float = None, n_trials: int = 20, mesh_size: float = None,
                              start: Dict[str, float] = None, out_name: str = None,
                              ctx: Context = None) -> Dict[str, Any]:
    """Find the lightest dimensions of a parametric model (ops/script as in create_model, numbers as
    param expressions) such that CalculiX stress <= yield/safety_factor (and displacement <=
    max_displacement mm if set). bounds: {"t":[2,12]} | {"t":[2,12,0.5]} (step) | {"n":{"choices":[..]}}.
    params: fixed values for the other names. Optuna TPE, one FEM per trial (~3-20 s each).
    Selectors must stay valid for all parameter values (prefer side/normal/near over face ids).
    out_name stores the best design as a model."""
    return await _run("optimize_dimensions", dict(
        bounds=bounds, fixed=fixed, loads=loads, ops=ops, script=script, params=params, material=material,
        safety_factor=safety_factor, stress_metric=stress_metric, max_displacement=max_displacement,
        n_trials=n_trials, mesh_size=mesh_size, start=start, out_name=out_name), ctx)


@mcp.tool
async def optimize_mesh(name: str = None, file: str = None, target_faces: int = None, reduction: float = None,
                        fill_holes: float = None, min_component_area: float = None, repair: bool = True,
                        to_solid: bool = True, out_name: str = None, device: str = None,
                        show: bool = True) -> ToolResult:
    """3D model optimiser (CuMesh on the GPU): repair and decimate a triangle mesh - a stored model
    (tessellated) or a mesh file (STL/OBJ/PLY/OFF). Repairs degenerate/duplicate faces, non-manifold
    edges, islands smaller than min_component_area (mm2, default 0.1 % of the area), holes up to
    fill_holes (max hole perimeter, mm) and face winding; then QEM-decimates to target_faces
    (or by reduction, e.g. 0.8 = remove 80 % of the faces). Writes work/exports/<out_name>.stl and,
    if the result is watertight, stores it as a solid model for analyze/optimize_topology.
    Reports before/after face counts, watertightness, area and volume. device: auto | gpu | cpu."""
    if name is None and file is None:
        raise ValueError("give name (stored model) or file (mesh path)")
    r = await _run("optimize_mesh", dict(name=name, file=file, target_faces=target_faces, reduction=reduction,
                                         fill_holes=fill_holes, min_component_area=min_component_area,
                                         repair=repair, to_solid=to_solid, out_name=out_name, device=device,
                                         render=show))
    return _with_previews(r, show)


@mcp.tool
async def export_web(name: str, path: str = None, compress: str = "cc", simplify: float = None,
                     simplify_error: float = None, aggressive: bool = False, quantize: bool = True,
                     position_bits: int = None, target_faces: int = None, result_fcstd: str = None,
                     deflection: float = None, units: str = "m", device: str = None) -> Dict[str, Any]:
    """Optimised .glb for web/AR/three.js viewers via gltfpack (meshoptimizer).
    - compress: cc (meshopt, smallest) | c | fallback (uncompressed fallback for old loaders) | none
    - simplify: triangle ratio 0..1 (gltfpack -si; simplify_error limits deviation, default 1 %;
      aggressive ignores quality), quantize=False disables KHR_mesh_quantization, position_bits 1-16.
    - target_faces: GPU pre-decimation with CuMesh before gltfpack (huge tessellations).
    - result_fcstd (from analyze/verify): vertex colours = von Mises stress (turbo colormap).
    - units: m (glTF standard, mm values scaled by 0.001) | mm.
    Returns raw vs packed bytes and the gltfpack report path."""
    return await _run("export_web", dict(name=name, path=path, compress=compress, simplify=simplify,
                                         simplify_error=simplify_error, aggressive=aggressive, quantize=quantize,
                                         position_bits=position_bits, target_faces=target_faces,
                                         result_fcstd=result_fcstd, deflection=deflection, units=units,
                                         device=device))


@mcp.tool
async def export_model(name: str, format: str = "step", path: str = None) -> Dict[str, Any]:
    """Export a stored model: step | iges | brep | stl | fcstd | glb (gltfpack-optimised, see export_web).
    Default path work/exports/<name>.<ext>."""
    return await _run("export_model", dict(name=name, format=format, path=path))


@mcp.tool
async def render_model(name: str, result_fcstd: str = None) -> ToolResult:
    """Preview image of a model (3 views); with result_fcstd (from analyze) coloured by von Mises."""
    r = await _run("render", dict(name=name, result_fcstd=result_fcstd))
    return _with_previews(r, True)


@mcp.tool
async def delete_model(name: str) -> Dict[str, Any]:
    """Delete a stored model."""
    return await _run("delete_model", dict(name=name))


def run_server(transport: str | None = None) -> None:
    selected = transport or os.getenv("FREECAD_MCP_TRANSPORT", "streamable-http")
    if selected in {"streamable-http", "http"}:
        stateless = os.getenv("FREECAD_MCP_STATELESS", "1") != "0"
        mcp.run(transport="http", path=SERVER_PATH, host=SERVER_HOST, port=SERVER_PORT, stateless_http=stateless)
    else:
        mcp.run(transport=selected)


if __name__ == "__main__":
    run_server(sys.argv[1] if len(sys.argv) > 1 else None)
