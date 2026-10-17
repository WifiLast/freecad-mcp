"""MCP test for gltfpack: export_web, export_model(glb), optimize_mesh auto-glb."""
import asyncio, json, sys
from fastmcp import Client
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:2022/freecad"

async def main():
    async with Client(URL, timeout=600) as c:
        print("export_web" in [t.name for t in await c.list_tools()])
        r = (await c.call_tool("export_web", {"name": "cantilever_cumesh80", "compress": "cc"})).data
        print("export_web", {k: r["gltfpack"][k] for k in ("raw_bytes", "packed_bytes", "size_ratio")})
        r = (await c.call_tool("export_model", {"name": "cantilever_cumesh80", "format": "glb"})).data
        print("export_model glb", r["gltfpack"]["glb"], r["gltfpack"]["packed_bytes"])
        o = await c.call_tool("optimize_mesh", {"file": "/home/wiffzack/mcp/freecad/work/test_scan.stl", "target_faces": 4000, "show": False})
        print("optimize_mesh glb", json.loads(o.content[0].text).get("glb"))
asyncio.run(main())
