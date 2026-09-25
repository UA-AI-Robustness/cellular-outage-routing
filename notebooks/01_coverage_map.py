"""Coverage map on three basemap styles for comparison.
Produces three PDFs; pick the one you like best."""
import osmnx as ox
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from connroute.config import load_config
from connroute.signal.layer import cache_path
from connroute.viz.style import GREEN, RED, save

cfg = load_config()
Gp = ox.load_graphml(cache_path(cfg))
for _, _, d in Gp.edges(data=True):
    d["dead_fraction"] = float(d["dead_fraction"])
for _, nd in Gp.nodes(data=True):
    nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])

# build colored edge segments once (reused for all three)
cmap = LinearSegmentedColormap.from_list("cov", [GREEN, "#F4D03F", RED])
segments, colors = [], []
for u, v, k, d in Gp.edges(keys=True, data=True):
    geom = d.get("geometry")
    if geom is not None:
        xs, ys = geom.xy
        pts = list(zip(xs, ys))
    else:
        pts = [(Gp.nodes[u]["x"], Gp.nodes[u]["y"]),
               (Gp.nodes[v]["x"], Gp.nodes[v]["y"])]
    segments.append(pts)
    colors.append(cmap(d["dead_fraction"]))


def render(source, attribution, name):
    fig, ax = plt.subplots(figsize=(7, 8))
    lc = LineCollection(segments, colors=colors, linewidths=0.7, zorder=3)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_axis_off()
    try:
        cx.add_basemap(ax, crs="EPSG:32610", source=source,
                       attribution=attribution, attribution_size=4)
        print(f"[ok] {name}")
    except Exception as e:
        print(f"[FAILED] {name}: {e}")
        plt.close(fig)
        return
    save(name, category="coverage", fig=fig)
    plt.close(fig)


# --- Option 1: OpenStreetMap Mapnik (colorful, guaranteed keyless) ---
render(cx.providers.OpenStreetMap.Mapnik,
       "© OpenStreetMap contributors",
       "coverage_map_osm")

# --- Option 2: CartoDB Positron light/grey (clean, best for overlay) ---
render("https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key={os.environ.get('CARTO_API_KEY', '')}",
       "© OpenStreetMap contributors © CARTO",
       "coverage_map_carto_light")

# --- Option 3: CartoDB Voyager (soft color, keyless CDN) ---
render("https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
       "© OpenStreetMap contributors © CARTO",
       "coverage_map_carto_voyager")

print("\nDone. Compare the three PDFs in results/figures/coverage/:")
print("  coverage_map_osm_<ts>.pdf            (OSM Mapnik, colorful)")
print("  coverage_map_carto_light_<ts>.pdf    (CARTO light, grey)")
print("  coverage_map_carto_voyager_<ts>.pdf  (CARTO Voyager, soft color)")