# experiments/exp_orderings_map.py
"""Five-route comparison (folium) on N different OD pairs: fastest, shortest,
continuity, call-adequate, upload-adequate. Routes drawn back-to-front with
decreasing width so overlapping routes remain visible as color bands. Legend,
dead-zone heat, CartoDB basemap. Each run writes to a timestamped subfolder.

    export CARTO_API_KEY="your_key"    # or set via conda env
    python -m experiments.exp_orderings_map
"""
from __future__ import annotations
import os, time, random
from datetime import datetime
from pathlib import Path
import numpy as np
import osmnx as ox
import folium
from folium.plugins import HeatMap

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.graph.build import cache_path as graph_cache_path
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route, constrained_route

CARTO_KEY = os.environ.get("CARTO_API_KEY", "")

MIN_OD_METERS = 3000.0
BUDGET = 150.0
N_PAIRS = 10                      # number of different OD pairs / figures

# draw order: back (widest) -> front (narrowest); overlaps show as nested bands
# each entry: (name, color, linewidth)
ROUTE_STYLE = [
    ("Fastest",         "#6C8EBF", 11),   # blue,  back, widest
    ("Shortest",        "#666666", 9),    # grey
    ("Continuity",      "#B85450", 7),    # red
    ("Call-adequate",   "#D6B656", 5),    # amber
    ("Upload-adequate", "#2CA02C", 3),    # green, front, narrowest
]
COLORS = {name: c for name, c, _ in ROUTE_STYLE}   # for the legend

# timestamped output subfolder
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FIGDIR = Path("results/figures/routes") / f"orderings_{STAMP}"
FIGDIR.mkdir(parents=True, exist_ok=True)


# ---- HTML -> PNG -> PDF ----
def _html_to_png(html_path, png_path, width=1100, height=1000, max_wait=40.0):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
    except ImportError:
        print("[pdf] selenium missing; HTML only"); return False
    opts = Options()
    for a in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
              "--disable-gpu", "--force-device-scale-factor=2",
              f"--window-size={width},{height}"):
        opts.add_argument(a)
    try:
        drv = webdriver.Chrome(options=opts)
    except Exception as e:
        print(f"[pdf] chromedriver unavailable ({e}); HTML only"); return False
    try:
        drv.set_page_load_timeout(90)
        drv.get("file://" + str(Path(html_path).resolve()))
        try: drv.execute_script("window.dispatchEvent(new Event('resize'));")
        except Exception: pass
        deadline = time.time() + max_wait; prev, stable, saw = -1, 0, False
        while time.time() < deadline:
            try:
                done, total = drv.execute_script(
                    "var t=document.querySelectorAll('img.leaflet-tile');var n=0;"
                    "for(var i=0;i<t.length;i++){if(t[i].complete&&t[i].naturalWidth>0)n++;}"
                    "return [n,t.length];")
            except Exception: done, total = 0, 0
            if total > 0: saw = True
            if total > 0 and done >= total and done == prev:
                stable += 1
                if stable >= 4: break
            else: stable = 0
            prev = done; time.sleep(0.5)
        time.sleep(3.0 if saw else 6.0)
        try:
            el = drv.find_element(By.CLASS_NAME, "folium-map")
            if el.size.get("height", 0) > 0: el.screenshot(str(png_path))
            else: drv.save_screenshot(str(png_path))
        except Exception: drv.save_screenshot(str(png_path))
        return True
    finally:
        drv.quit()

def _png_to_pdf(png_path, pdf_path):
    try:
        from PIL import Image
    except ImportError: return False
    try:
        Image.open(png_path).convert("RGB").save(pdf_path, "PDF", resolution=200.0); return True
    except Exception: return False

def export(html_path, stem):
    png = FIGDIR / f"{stem}.png"
    if _html_to_png(html_path, png): _png_to_pdf(png, FIGDIR / f"{stem}.pdf")


# ---- graphs ----
cfg = load_config()
Gp = ox.load_graphml(cache_path(cfg)); _coerce_types(Gp)
for _, _, dta in Gp.edges(data=True):
    for a in ("d_dead", "d_lowrate", "d_lowupload", "time", "length", "dead_fraction"):
        if a in dta: dta[a] = float(dta[a])
for _, nd in Gp.nodes(data=True): nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])
G = ox.load_graphml(graph_cache_path(cfg))

def latlon(n):
    nd = G.nodes[n]; return (float(nd["y"]), float(nd["x"]))
def sl(s, d):
    return float(np.hypot(Gp.nodes[s]["x"] - Gp.nodes[d]["x"],
                          Gp.nodes[s]["y"] - Gp.nodes[d]["y"]))

heat_pts = []
for u, v, k, dta in Gp.edges(keys=True, data=True):
    df = dta["dead_fraction"]
    if df <= 0: continue
    (la1, lo1), (la2, lo2) = latlon(u), latlon(v)
    heat_pts.append([0.5 * (la1 + la2), 0.5 * (lo1 + lo2), df])

prefs = build_preferences(cfg)

def five_routes(s, d):
    routes = {}
    routes["Fastest"], _ = lexico_route(Gp, s, d, make_order(prefs, ["time"]))
    routes["Shortest"], _ = lexico_route(Gp, s, d, make_order(prefs, ["distance"]))
    for name, attr in [("Continuity", "d_dead"), ("Call-adequate", "d_lowrate"),
                       ("Upload-adequate", "d_lowupload")]:
        rp, _, _ = constrained_route(Gp, s, d, budget_attr=attr, budget=BUDGET, cost_attr="time")
        routes[name] = rp
    return routes

def n_distinct(routes):
    seen = set()
    for p in routes.values():
        if p: seen.add(tuple(p))
    return len(seen)


def legend_html():
    rows = "".join(
        f"<div style='margin:2px 0'><span style='display:inline-block;width:18px;"
        f"height:4px;background:{c};margin-right:6px;vertical-align:middle'></span>"
        f"{name}</div>"
        for name, c in COLORS.items()
    )
    return (
        "<div style='position:fixed;bottom:20px;left:20px;z-index:9999;"
        "background:white;padding:8px 12px;border:1px solid #888;border-radius:4px;"
        "font:13px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,0.3)'>"
        "<b>Routes</b>" + rows + "</div>"
    )


def make_map(s, d, routes, idx):
    m = folium.Map(location=latlon(s), zoom_start=13, tiles=None, width=1100, height=900)
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=" + CARTO_KEY,
        attr="© OpenStreetMap contributors © CARTO", name="CartoDB Voyager").add_to(m)
    HeatMap(heat_pts, radius=12, blur=18, min_opacity=0.3, name="Dead-zone density").add_to(m)
    # draw back-to-front with decreasing width so overlapped routes stay visible
    for name, color, lw in ROUTE_STYLE:
        path = routes.get(name)
        if not path: continue
        folium.PolyLine([latlon(n) for n in path], color=color, weight=lw,
                        opacity=0.9, tooltip=name).add_to(m)
    folium.Marker(latlon(s), tooltip="S", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(latlon(d), tooltip="D", icon=folium.Icon(color="red")).add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html()))
    folium.LayerControl(collapsed=False).add_to(m)
    html = FIGDIR / f"orderings_pair{idx}.html"; m.save(str(html))
    print(f"saved {html}")
    export(html, f"orderings_pair{idx}")


if __name__ == "__main__":
    rng = random.Random()          # no seed -> different pairs every run
    nodes = list(Gp.nodes())
    found = 0; tries = 0
    print(f"output folder: {FIGDIR}")
    while found < N_PAIRS and tries < 40000:
        tries += 1
        s, d = rng.choice(nodes), rng.choice(nodes)
        if s == d or sl(s, d) < MIN_OD_METERS: continue
        routes = five_routes(s, d)
        if not all(routes.values()):        # need all five feasible
            continue
        if n_distinct(routes) < 3:           # need visibly different routes
            continue
        found += 1
        print(f"pair {found}: {s}->{d}  ({n_distinct(routes)} distinct routes)")
        make_map(s, d, routes, found)
    if found == 0:
        print("no suitable OD pairs; loosen BUDGET / MIN_OD_METERS / distinct threshold")
    else:
        print(f"\nproduced {found} five-route maps in {FIGDIR}")