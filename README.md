# FreeCAD MCP – simulate forces, remove material

MCP server (FastMCP, streamable HTTP, default `http://0.0.0.0:2022/freecad`) that builds or imports
parts in FreeCAD, simulates static loads (FreeCAD FEM + gmsh + CalculiX) and reduces material:

- **optimize_topology** – SIMP topology optimisation on a voxel grid inside the part. The result is a
  smoothed iso-surface trimmed with the original solid, so mounting faces and holes stay exact.
  Exported as STL + STEP, then re-checked with CalculiX against the original.
  - `shell` (mm): 3D-print mode. A solid skin of that thickness is kept on every surface and only
    the interior is optimised. For mesh sources (STL/OBJ/PLY/OFF) the exported STL is the source mesh
    plus the cavities as inner shells, cut on a fine grid (voxel = shell/2, skin measured from the
    full-resolution surface), so no boolean with the often invalid mesh solid is needed.
  - `quality`: `draft` (default: 25 iterations, loose solves) | `normal` | `fine`.
  - `minify` / `minify_error` (default 0.05 mm): last step, error-bounded simplification
    (meshoptimizer via gltfpack). Evelynn figure: 634k → 205k triangles, mean deviation 0.02 mm.
  - Example: 270 mm figure (500k-triangle STL), 2 mm skin, resolution 100 → about 45 s end to end.
- **optimize_dimensions** – lightest values of a parametric model's dimensions under a stress
  (and optional deflection) limit; Optuna TPE with one CalculiX run per trial.

- **optimize_mesh** – 3D model optimiser on the GPU (CuMesh): repairs degenerate/duplicate faces,
  non-manifold edges, small islands, holes and winding, then QEM-decimates. Works on a stored model
  or a mesh file (STL/OBJ/PLY/OFF) and stores watertight results as solids.

- **export_web** – web/AR-ready `.glb` via gltfpack (meshoptimizer). It writes metres, applies
  quantisation and meshopt compression, and can simplify the mesh, pre-decimate it with CuMesh and
  colour it by stress.

Units: mm, N, MPa, g. Tools: `list_models, create_model, inspect_model, analyze, optimize_topology,
optimize_dimensions, optimize_mesh, export_web, export_model, render_model, delete_model`.

## Layout

| file | role |
|---|---|
| `freecad_mcp.py` | MCP server (fc env); one `fc_worker.py` process per job, log lines streamed as progress |
| `fc_worker.py` | job actions: models, FEM, voxelisation, BC mapping, SIMP driver, mesh→solid, verify |
| `fc_core.py` | FreeCAD helpers: CSG ops, import, store (`work/models/*.brep+json`), selectors, materials |
| `fem_ccx.py` | static analysis with FreeCAD FEM (gmsh 2nd-order tets, CalculiX) |
| `topopt.py` | numpy hex8 voxel FEM, SIMP/OC, density filter, symmetry, CPU (CHOLMOD) + GPU client |
| `gpu_solver.py` | torch GPU helper (py310_amd, ROCm): matrix-free Jacobi-PCG, batched load cases, mesh commands |
| `mesh_gpu.py` | CuMesh repair + decimation, runs inside the GPU helper |
| `third_party/CuMesh` | ROCm build of CuMesh (not in git; `patches/cumesh-rocm-setup.patch` = the C++20 switch for torch 2.15) |
| `other/` | upstream reference checkouts (FreeCAD, torch-fem, CuMesh, GPU FEM) - not in git |

## GPU acceleration

Voxel solves run on the GPU (RX 7900 XTX, ROCm 7.2 torch in `py310_amd`). The approach follows
`other/torch-fem` (torch sparse/iterative solves on the device) and
`other/GPU-accelerated-Finite-Element-Method-using-Python-and-CUDA` (batched GPU element work).
Because the grid is regular, no global matrix is assembled: `K·u = scatter(E_e · KE @ u[edof])`
with a single 24×24 KE. CG is warm-started across SIMP iterations, and SIMP solves use tol 1e-3
(draft) to 1e-5 (fine); the final stress evaluation uses 1e-6.

The CG preconditioner is a geometric multigrid V-cycle. Each coarse voxel is exactly 8 fine voxels,
so the Galerkin coarse element matrices are `Σ_children Q_oᵀ K_child Q_o` with 8 fixed 24×24
prolongations (exact `PᵀKP`, matrix-free, rebuilt every solve). Damped-Jacobi smoothing uses a
per-level ω from a power-iteration estimate of λmax(D⁻¹A), which is about 4-6 here, so a fixed ω=0.6
diverges. The coarsest level is factored densely. Evelynn figure, 93.5k voxels, 2 load cases:
about 105 CG iterations / 1.6-3 s per solve, against 13-19k iterations / 18-26 s with Jacobi.
`FREECAD_GPU_MG=0` switches back to Jacobi. Voxels joined only by an edge or corner (zero-stiffness
hinges) are dropped before the solve.

Voxelisation is a scanline even-odd fill of the tessellation along x, y and z with a 2-of-3 vote
(1.5 s for 20k triangles). It falls back to OCC ray booleans when the voxel volume is far off the
solid volume.

The server starts one shared helper (`work/gpu.sock`) on the first topology job, so later jobs skip
torch/ROCm start-up. The helper exits after `FREECAD_GPU_IDLE` s (default 600) to give VRAM back
to ollama. `device`: `auto` (GPU above 4000 voxels, falls back to CPU) | `gpu` | `cpu`.

Cantilever 120×30×40 mm, 30 % volume, 60 SIMP iterations:

| resolution | voxels | CPU CHOLMOD | GPU |
|---|---|---|---|
| 40 | 5.6k | 74 s total (SIMP 39 s) | 15 s total (SIMP 7 s) |
| 80 | 43k | ~7 min (est.) | 34 s total (SIMP 22 s) |

Single solve: 36k voxels 7.3 s CPU vs 0.27 s GPU; 120k voxels 1.0 s GPU.
CalculiX (verification and `analyze`) stays on the CPU.

## CuMesh (GPU mesh optimisation)

The HIP port of CuMesh (`https://github.com/Cardboard-box-a/CuMesh`, commit 2ef5f9d) is built in a
copy, `third_party/CuMesh`, so hipify cannot overwrite the hand-ported `.hip` files in the original:

    git clone https://github.com/Cardboard-box-a/CuMesh third_party/CuMesh
    cd third_party/CuMesh && git checkout 2ef5f9d && git submodule update --init --recursive
    git apply ../../patches/cumesh-rocm-setup.patch                    # torch 2.15 needs C++20
    ROCM_PATH=/opt/rocm/core-7.14 GPU_ARCHS=gfx1100 MAX_JOBS=6 PYTHONNOUSERSITE=1 \
        /mnt/data/miniconda3/envs/py310_amd/bin/python setup.py build_ext --inplace

Where it is used:
- `optimize_mesh`.
- Surface of `optimize_topology`: marching cubes → Taubin smoothing → CuMesh repair/decimation →
  solid → trimmed with the original.
- Mesh imports in `create_model` (decimated above `mesh_max_faces`, default 20000).

Each step falls back to FreeCAD's CPU mesh tools if CuMesh is unavailable. If a GPU-decimated
topology surface gives an invalid solid, that job also falls back to CPU decimation.

Two fixes are needed for valid solids:
- CuMesh can leave distinct vertices at identical positions where sheets touch.
  `separate_coincident` pushes them 1e-3 voxel apart.
- `mesh_to_solid` builds one shell per connected component with a tiny sewing tolerance.

Speed: a 328k-face sphere → 6k faces takes 0.36 s on the GPU vs 3.5 s on the CPU (FreeCAD), both
watertight. A topology surface of 35k → 5.8k faces takes 0.08 s.

## gltfpack (.glb optimisation)

Installed as `npm install -g --prefix ~/.local gltfpack`: plain `-g` needs root for `/usr/local`.
The binary is found via `$GLTFPACK`, then `PATH`, then `~/.local/bin`, then `/usr/local/bin`.

`pack_glb()` writes the mesh with trimesh (mm → m, optional vertex colours) and then runs
`gltfpack -cc` (KHR_mesh_quantization + EXT_meshopt_compression), with optional `-si/-se/-sa` for
simplification. `-cf` adds an uncompressed fallback; `-noq` writes plain glTF.

Where it is used:
- `export_web` and `export_model(format="glb")`.
- Automatically for `optimize_topology` (`exports.glb`) and `optimize_mesh` (`glb`).

Packed files decode with three.js/Babylon (they need the meshopt decoder for `-cc`).

Results:
- Optimised cantilever, 3.9k triangles: 70 KB → 14 KB (-80 %); with `simplify=0.5`: 7.9 KB.
- Stress-coloured part, 26k triangles: 520 KB → 91 KB.

## Environment

`~/miniforge3/envs/fc`: FreeCAD 26.3, gmsh, CalculiX, scipy, scikit-image, scikit-sparse,
`lazy-loader<0.6`, pip fastmcp/trimesh/optuna/pyamg.

Quirks:
- FreeCAD's init deletes names already defined in `__main__`, so import FreeCAD first.
- FreeCAD's `Ext/lazy_loader` shadows the pip package that scikit-image needs (see `import_skimage_measure`).
- A force direction needs a helper edge (`Direction`); setting `DirectionVector` is ignored.

## Run

    ~/miniforge3/envs/fc/bin/python freecad_mcp.py            # foreground
    sudo ./install_freecad_mcp_service.sh                      # systemd unit freecad-mcp
    ~/miniforge3/envs/fc/bin/python tests/quick_mcp_test.py    # smoke test
    ~/miniforge3/envs/fc/bin/python tests/bench_gpu.py         # CPU vs GPU solve benchmark
