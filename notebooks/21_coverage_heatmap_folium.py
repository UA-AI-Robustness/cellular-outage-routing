# notebooks/21_coverage_heatmap_folium.py
"""Dead-zone density heatmap (folium), ACM style: blue->green HeatMap on a
CartoDB basemap. No route overlay, no S/D markers. Saves HTML, then exports
PNG + PDF via headless Chrome (selenium) + Pillow, mirroring the ACM pipeline.

    export CARTO_API_KEY="your_key_here"
    python notebooks/21_coverage_heatmap_folium.py

PDF export needs: selenium + a chromedriver on PATH, and Pillow.
    conda install -c conda-forge selenium python-chromedriver-binary pillow
(If selenium/chromedriver are unavailable, the HTML is still written; open it
and Print -> Save as PDF as a fallback.)
"""
import os
import time
from pathlib import Path
import osmnx as ox
import folium
from folium.plugins import HeatMap

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.graph.build import cache_path as graph_cache_path

FIGDIR = Path("results/figures/coverage")
FIGDIR.mkdir(parents=True, exist_ok=True)

CARTO_KEY = os.environ.get("CARTO_API_KEY", "")   # set in terminal, NOT in code


# ---------------------------------------------------------------------------
# HTML -> PNG -> PDF export (adapted from the ACM make_figures pipeline)
# ---------------------------------------------------------------------------
def _html_to_png(html_path, png_path, width=1200, height=1000, max_wait=40.0):
    """Render a folium HTML map to PNG with headless Chrome, waiting for the
    Leaflet tiles and heat canvas to finish loading."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
    except ImportError:
        print("[pdf] selenium not installed; PNG/PDF export skipped "
              "(open the HTML and Print -> Save as PDF)")
        return False
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--force-device-scale-factor=2")   # crisp 2x
    opts.add_argument(f"--window-size={width},{height}")
    try:
        drv = webdriver.Chrome(options=opts)
    except Exception as e:
        print(f"[pdf] chromedriver unavailable ({e}); PNG/PDF export skipped")
        return False
    try:
        drv.set_page_load_timeout(90)
        drv.get("file://" + str(Path(html_path).resolve()))
        try:
            drv.execute_script("window.dispatchEvent(new Event('resize'));")
        except Exception:
            pass
        # wait until tiles are loaded and stable
        deadline = time.time() + max_wait
        prev_done, stable, saw = -1, 0, False
        while time.time() < deadline:
            try:
                done, total = drv.execute_script(
                    "var t=document.querySelectorAll('img.leaflet-tile');"
                    "var n=0;for(var i=0;i<t.length;i++){"
                    "  if(t[i].complete && t[i].naturalWidth>0) n++;}"
                    "return [n, t.length];")
            except Exception:
                done, total = 0, 0
            if total > 0:
                saw = True
            if total > 0 and done >= total and done == prev_done:
                stable += 1
                if stable >= 4:
                    break
            else:
                stable = 0
            prev_done = done
            time.sleep(0.5)
        time.sleep(4.0 if saw else 8.0)   # let the heat canvas paint
        try:
            el = drv.find_element(By.CLASS_NAME, "folium-map")
            if el.size.get("height", 0) > 0 and el.size.get("width", 0) > 0:
                el.screenshot(str(png_path))
            else:
                drv.save_screenshot(str(png_path))
        except Exception:
            drv.save_screenshot(str(png_path))
        return True
    finally:
        drv.quit()


def _png_to_pdf(png_path, pdf_path):
    """Wrap a hi-res PNG into a single-page PDF (Pillow)."""
    try:
        from PIL import Image
    except ImportError:
        print("[pdf] Pillow not installed; PNG->PDF wrap skipped")
        return False
    try:
        Image.open(png_path).convert("RGB").save(pdf_path, "PDF", resolution=200.0)
        return True
    except Exception as e:
        print(f"[pdf] PNG->PDF wrap failed ({e})")
        return False


def export_html_to_pdf(html_path, stem):
    png = FIGDIR / f"{stem}.png"
    if _html_to_png(html_path, png):
        pdf = FIGDIR / f"{stem}.pdf"
        if _png_to_pdf(png, pdf):
            print(f"saved {png}\nsaved {pdf}")


# ---------------------------------------------------------------------------
# Build the heatmap
# ---------------------------------------------------------------------------
cfg = load_config()
Gp = ox.load_graphml(cache_path(cfg)); _coerce_types(Gp)   # projected, dead_fraction
G = ox.load_graphml(graph_cache_path(cfg))                 # unprojected, lat/lon

deadf = {}
for u, v, k, d in Gp.edges(keys=True, data=True):
    deadf[(u, v, k)] = float(d["dead_fraction"])


def latlon(n):
    nd = G.nodes[n]
    return (float(nd["y"]), float(nd["x"]))


heat_pts = []
for u, v, k, d in G.edges(keys=True, data=True):
    df = deadf.get((u, v, k), 0.0)
    if df <= 0:
        continue
    (la1, lo1), (la2, lo2) = latlon(u), latlon(v)
    heat_pts.append([0.5 * (la1 + la2), 0.5 * (lo1 + lo2), df])

lats = [float(nd["y"]) for _, nd in G.nodes(data=True)]
lons = [float(nd["x"]) for _, nd in G.nodes(data=True)]
center = (sum(lats) / len(lats), sum(lons) / len(lons))

m = folium.Map(location=center, zoom_start=13, tiles=None,
               width=1100, height=900)
folium.TileLayer(
    tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=" + CARTO_KEY,
    attr="© OpenStreetMap contributors © CARTO",
    name="CartoDB Voyager",
).add_to(m)
HeatMap(heat_pts, radius=12, blur=18, min_opacity=0.3,
        name="Dead-zone density").add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

out_html = FIGDIR / "coverage_heatmap.html"
m.save(str(out_html))
print(f"saved {out_html}   ({len(heat_pts)} weighted dead-edge points)")

# export to PNG + PDF
export_html_to_pdf(out_html, "coverage_heatmap")