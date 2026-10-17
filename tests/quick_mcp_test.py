"""Short MCP smoke test: list tools, small topology run on the GPU (no CalculiX verify)."""
import asyncio
import json
import sys
import time

from fastmcp import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:2022/freecad"


async def main():
    async with Client(URL, timeout=900) as c:
        print("tools:", [t.name for t in await c.list_tools()])
        await c.call_tool("create_model", {"name": "cantilever", "ops": [{"box": [120, 30, 40]}], "material": "aluminium",
                                           "list_faces": False})
        t = time.time()
        o = await c.call_tool("optimize_topology", {"name": "cantilever", "fixed": ["xmin"],
                                                    "loads": [{"faces": "xmax", "force": [0, 0, -2000]}],
                                                    "volume_fraction": 0.3, "resolution": 40, "verify": False,
                                                    "out_name": "cantilever_mcp"})
        r = json.loads(o.content[0].text)
        print(round(time.time() - t, 1), "s", r["solver"], r["optimized"]["material_saved_pct"],
              "images:", sum(getattr(x, "type", "") == "image" for x in o.content))


asyncio.run(main())
