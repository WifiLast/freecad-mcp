"""End-to-end test through the MCP HTTP endpoint: L-bracket in PLA, analyze + topology (SF search)."""
import asyncio, json, sys, time
from fastmcp import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:2022/freecad"

BRACKET = [
    {"box": [60, 40, 6]},                                   # horizontal arm (z 0..6)
    {"box": [6, 40, 50], "at": [0, 0, 0]},                  # vertical wall at x 0..6
    {"cylinder": {"r": 3.2, "h": 10}, "at": [-2, 10, 35], "axis": "x", "cut": True},   # wall bolt holes
    {"cylinder": {"r": 3.2, "h": 10}, "at": [-2, 30, 35], "axis": "x", "cut": True},
    {"cylinder": {"r": 4, "h": 10}, "at": [50, 20, -2], "cut": True},                  # load hole
]


def text(res):
    return json.loads(res.content[0].text) if res.content and hasattr(res.content[0], "text") else res.data


async def main():
    async with Client(URL, timeout=3600) as c:
        tools = await c.list_tools()
        print("tools:", [t.name for t in tools])
        t = time.time()
        r = text(await c.call_tool("create_model", {"name": "bracket", "ops": BRACKET, "material": "pla"}))
        print("create", round(time.time() - t, 1), r["volume_mm3"], r["mass_g"], len(r["faces"]))
        holes = text(await c.call_tool("inspect_model", {"name": "bracket", "list_faces": False,
                                                         "select": {"type": "cylinder", "radius": 3.2}}))
        print("wall holes:", holes["selected_faces"])
        fixed = [{"type": "cylinder", "radius": 3.2}]
        loads = [{"faces": {"type": "cylinder", "radius": 4}, "force": [0, 0, -150]}]
        t = time.time()
        a = text(await c.call_tool("analyze", {"name": "bracket", "fixed": fixed, "loads": loads, "material": "pla"}))
        print("analyze", round(time.time() - t, 1), a["von_mises_MPa"], a["displacement_mm"]["max"], a["safety_factor"])
        t = time.time()
        o = await c.call_tool("optimize_topology", {"name": "bracket", "fixed": fixed, "loads": loads, "material": "pla",
                                                    "target_safety_factor": 2.0, "resolution": 45, "symmetry": ["y"]})
        r = text(o)
        print("topo", round(time.time() - t, 1), "images", sum(1 for x in o.content if getattr(x, "type", "") == "image"))
        print(json.dumps({k: r[k] for k in ("original", "optimized", "search", "verify_calculix") if k in r}, indent=1)[:4000])

asyncio.run(main())
