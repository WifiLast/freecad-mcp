"""MCP test for CuMesh integration: optimize_mesh on an STL, then a topology run (GPU surface decimation)."""
import asyncio, json, sys, time
from fastmcp import Client
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:2022/freecad"

async def main():
    async with Client(URL, timeout=900) as c:
        t = time.time()
        o = await c.call_tool("optimize_mesh", {"file": "/home/wiffzack/mcp/freecad/work/test_scan.stl", "target_faces": 4000})
        r = json.loads(o.content[0].text)
        print("optimize_mesh", round(time.time() - t, 1), "s", r["cumesh"]["backend"], r["cumesh"]["input"], "->",
              r["cumesh"]["output"], r["after"], r.get("model"), r.get("solid_note"))
        t = time.time()
        o = await c.call_tool("optimize_topology", {"name": "cantilever", "fixed": ["xmin"],
              "loads": [{"faces": "xmax", "force": [0, 0, -2000]}], "volume_fraction": 0.3,
              "resolution": 40, "verify": False, "out_name": "cantilever_cumesh", "target_faces": 6000})
        r = json.loads(o.content[0].text)
        print("topology", round(time.time() - t, 1), "s", r["surface_mesh"], r["geometry"], r["optimized"]["material_saved_pct"])
asyncio.run(main())
