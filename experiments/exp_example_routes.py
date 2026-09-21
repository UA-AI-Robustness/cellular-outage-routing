"""Example-route figures (folium): fastest vs. budget-constrained connectivity
route, over a dead-zone heat layer on a CartoDB basemap. Produces N interactive
HTML maps and exports each to PNG + PDF (headless Chrome + Pillow).

    export CARTO_API_KEY="your_key_here"
    python -m experiments.exp_example_routes
"""
from __future__ import annotations
import os
import time
import random
import numpy as np
import osmnx as ox
import folium
from folium.plugins import HeatMap
from tqdm import tqdm

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.graph.build import cache_path as graph_cache_path
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route, constrained_route

from pathlib import Path
FIGDIR = Path("results/figures/routes")
FIGDIR.mkdir(parents=True, exist_ok=True)

MIN_OD_METERS = 2500.0
MIN_DEAD_REMOVED = 300.0
DEAD_BUDGET = 100.0
N_EXAMPLES = 10
CARTO_KEY = os.environ.get("CARTO_API_KEY", "cb1_3rgg_1_311949a0a459755f7ec307af")


# ---------------------------------------------------------------------------
# HTML -> PNG -> PDF (same pipeline as the heatmap script)
# ---------------------------------------------------------------------------
def _html_to_png(html_path, png_path, width=1100, height=1000, max_wait=40.0):
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
    except ImportError:
        print("[pdf] selenium missing; HTML only")
        return False
    opts = Options()
    for a in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
              "--disable-gpu", "--force-device-scale-factor=2",
              f"--window-size={width},{height}"):
        opts.add_argument(a)
    try:
        drv = webdriver.Chrome(options=opts)
    except Exception as e:
        print(f"[pdf] chromedriver unavailable ({e}); HTML only")
        return False
    try:
        drv.set_page_load_timeout(90)
        drv.get("file://" + str(Path(html_path).resolve()))
        try:
            drv.execute_script("window.dispatchEvent(new Event('resize'));")
        except Exception:
            pass
        deadline = time.time() + max_wait
        prev, stable, saw = -1, 0, False
        while time.time() < deadline:
            try:
                done, total = drv.execute_script(
                    "var t=document.querySelectorAll('img.leaflet-tile');var n=0;"
                    "for(var i=0;i<t.length;i++){if(t[i].complete&&t[i].naturalWidth>0)n++;}"
                    "return [n,t.length];")
            except Exception:
                done, total = 0, 0
            if total > 0:
                saw = True
            if total > 0 and done >= total and done == prev:
                stable += 1
                if stable >= 4:
                    break
            else:
                stable = 0
            prev = done
            time.sleep(0.5)
        time.sleep(3.0 if saw else 6.0)
        try:
            el = drv.find_element(By.CLASS_NAME, "folium-map")
            if el.size.get("height", 0) > 0:
                el.screenshot(str(png_path))
            else:
                drv.save_screenshot(str(png_path))
        except Exception:
            drv.save_screenshot(str(png_path))
        return True
    finally:
        drv.quit()


def _png_to_pdf(png_path, pdf_path):
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        Image.open(png_path).convert("RGB").save(pdf_path, "PDF", resolution=200.0)
        return True
    except Exception:
        return False


def export(html_path, stem):
    png = FIGDIR / f"{stem}.png"
    if _html_to_png(html_path, png):
        _png_to_pdf(png, FIGDIR / f"{stem}.pdf")


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------
cfg = load_config()
Gp = ox.load_graphml(cache_path(cfg)); _coerce_types(Gp)   # projected, has d_dead/dead_fraction
for _, _, dta in Gp.edges(data=True):
    for a in ("d_dead", "time", "length", "dead_fraction"):
        dta[a] = float(dta[a])
for _, nd in Gp.nodes(data=True):
    nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])

G = ox.load_graphml(graph_cache_path(cfg))                 # unprojected, lat/lon


def latlon(n):
    nd = G.nodes[n]
    return (float(nd["y"]), float(nd["x"]))


def straight_line_m(s, d):
    return float(np.hypot(Gp.nodes[s]["x"] - Gp.nodes[d]["x"],
                          Gp.nodes[s]["y"] - Gp.nodes[d]["y"]))


def path_dead_time(path):
    dd = tt = 0.0
    for a, b in zip(path[:-1], path[1:]):
        ed = min(Gp[a][b].values(), key=lambda e: float(e["time"]))
        dd += float(ed["d_dead"]); tt += float(ed["time"])
    return dd, tt


# dead-zone heat points (edge midpoints weighted by dead_fraction)
heat_pts = []
for u, v, k, dta in Gp.edges(keys=True, data=True):
    df = dta["dead_fraction"]
    if df <= 0:
        continue
    (la1, lo1), (la2, lo2) = latlon(u), latlon(v)
    heat_pts.append([0.5 * (la1 + la2), 0.5 * (lo1 + lo2), df])


def make_map(s, d, fast_path, conn_path, idx):
    m = folium.Map(location=latlon(s), zoom_start=13, tiles=None,
                   width=1100, height=900)
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=" + CARTO_KEY,
        attr="© OpenStreetMap contributors © CARTO", name="CartoDB Voyager",
    ).add_to(m)
    HeatMap(heat_pts, radius=12, blur=18, min_opacity=0.3,
            name="Dead-zone density").add_to(m)
    folium.PolyLine([latlon(n) for n in fast_path], color="#C45327", weight=5,
                    opacity=0.9, tooltip="Fastest").add_to(m)
    folium.PolyLine([latlon(n) for n in conn_path], color="#1a1a1a", weight=4,
                    opacity=0.95, tooltip="Connectivity-aware").add_to(m)
    folium.Marker(latlon(s), tooltip="S", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(latlon(d), tooltip="D", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    html = FIGDIR / f"example_route_{idx}.html"
    m.save(str(html))
    export(html, f"example_route_{idx}")


if __name__ == "__main__":
    prefs = build_preferences(cfg)
    base = make_order(prefs, ["time"])
    rng = random.Random(cfg.seed)
    nodes = list(Gp.nodes())
    found = tries = 0

    pbar = tqdm(total=N_EXAMPLES, desc="finding examples", unit="fig")
    while found < N_EXAMPLES and tries < 40000:
        tries += 1
        s, d = rng.choice(nodes), rng.choice(nodes)
        if s == d or straight_line_m(s, d) < MIN_OD_METERS:
            continue
        fast_path, _ = lexico_route(Gp, s, d, base)
        if not fast_path:
            continue
        fast_dead, fast_time = path_dead_time(fast_path)
        if fast_dead < MIN_DEAD_REMOVED:
            continue
        conn_path, ctime, cbud = constrained_route(Gp, s, d, budget_attr="d_dead",
                                                   budget=DEAD_BUDGET, cost_attr="time")
        if not conn_path or conn_path == fast_path:
            continue
        conn_dead, conn_time = path_dead_time(conn_path)
        detour = 100.0 * (conn_time - fast_time) / fast_time if fast_time else 0.0
        found += 1
        tqdm.write(f"example {found}: {s}->{d}  fast_dead={fast_dead:.0f}m -> "
                   f"conn_dead={conn_dead:.0f}m  detour={detour:.0f}%")
        make_map(s, d, fast_path, conn_path, found)
        pbar.update(1)
    pbar.close()
    print(f"\nproduced {found} folium example maps (HTML+PNG+PDF) in {FIGDIR}"
          if found else "no illustrative example found; loosen thresholds")