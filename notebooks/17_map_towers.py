# notebooks/17_map_towers.py
import osmnx as ox, matplotlib.pyplot as plt, contextily as cx
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.signal.towers import load_towers
from connroute.viz.style import apply_style, RED, save

apply_style(usetex=False)
cfg = load_config()
towers = load_towers("data/raw/opencellid_us_310.csv")   # already filtered to SF bbox

# towers.df has x,y in EPSG:32610 (UTM). contextily needs Web Mercator (3857) OR
# we pass the source CRS and let contextily reproject the basemap.
fig, ax = plt.subplots(figsize=(6, 7))
ax.scatter(towers.df["x"], towers.df["y"], s=6, c=RED, alpha=0.6,
           edgecolors="none", zorder=3, label=f"{len(towers.df)} LTE towers")
ax.set_axis_off()
cx.add_basemap(ax, crs="EPSG:32610",
               source=cx.providers.CartoDB.Positron, attribution_size=5)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
save("towers_map", category="towers", fig=fig)