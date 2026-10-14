"""Manual check that stepping the dashboard moves nothing: `pixi run -e dev python tests/manual/probe_dashboard_layout.py http://127.0.0.1:8050/`.

Steps eight instants in headless Chromium and reports every graph or details element whose position or
height changed while the view updated (expect: none, or a few px of tick-label wobble)."""
import sys, time
from playwright.sync_api import sync_playwright
URL = sys.argv[1]
JS = """() => { const out = {}; document.querySelectorAll('.dash-graph, details').forEach((el, i) => {
  const r = el.getBoundingClientRect(); out[el.id || ('details' + i)] = [Math.round(r.y), Math.round(r.height)]; }); return out; }"""
with sync_playwright() as p:
    b = p.chromium.launch(); page = b.new_page(viewport={"width": 2400, "height": 1500})
    page.goto(URL); page.wait_for_selector(".dash-graph .js-plotly-plot", timeout=60000); page.wait_for_timeout(2500)
    base = page.evaluate(JS); jumps = {}; samples = 0
    for step in range(8):
        page.click("#fwd"); t_end = time.time() + 1.0
        while time.time() < t_end:
            cur = page.evaluate(JS); samples += 1
            for k, v in cur.items():
                if k in base and v != base[k]:
                    jumps[k] = max(jumps.get(k, 0), max(abs(v[0] - base[k][0]), abs(v[1] - base[k][1])))
            page.wait_for_timeout(10)
    print(f"{URL}: {samples} samples over 8 steps; elements that moved or resized (max px):", jumps or "none")
    b.close()
