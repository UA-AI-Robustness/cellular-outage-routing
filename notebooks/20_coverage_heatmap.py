# notebooks/20_coverage_heatmap.py
"""Dead-zone density heatmap over SF, CartoDB basemap + hexbin, ACM style."""
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types

# ACM style
from pylab import rcParams
rcParams.update({"axes.labelsize": 9, "font.size": 9, "legend.fontsize": 7,
                 "text.usetex": False, "figure.figsize": [4.2, 4.6]})

cfg = load_config()
G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _, _, dta in G.edges(data=True):
    dta["dead_fraction"] = float(dta["dead_fraction"])
    dta["length"] = float(dta["length"])
for _, nd in G.nodes(data=True):
    nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])

# graph is UTM (EPSG:32610); contextily wants Web Mercator (3857)
to3857 = Transformer.from_crs("EPSG:32610", "EPSG:3857", always_xy=True)

# one weighted point per edge midpoint: weight = dead_fraction (× length so
# long dead stretches count more). Only edges with some dead exposure.
xs, ys, w = [], [], []
for u, v, k, dta in G.edges(keys=True, data=True):
    df = dta["dead_fraction"]
    if df <= 0:
        continue
    mx = 0.5 * (G.nodes[u]["x"] + G.nodes[v]["x"])
    my = 0.5 * (G.nodes[u]["y"] + G.nodes[v]["y"])
    X, Y = to3857.transform(mx, my)
    xs.append(X); ys.append(Y); w.append(df * dta["length"])

xs = np.array(xs); ys = np.array(ys); w = np.array(w)

fig, ax = plt.subplots(figsize=(6, 7))
hb = ax.hexbin(xs, ys, C=w, gridsize=60, cmap="inferno", mincnt=1,
               alpha=0.65, reduce_C_function=np.sum, linewidths=0, zorder=3)
ax.set_axis_off()

# frame to the data with a little padding
dx = (xs.max() - xs.min()) * 0.04; dy = (ys.max() - ys.min()) * 0.04
ax.set_xlim(xs.min() - dx, xs.max() + dx)
ax.set_ylim(ys.min() - dy, ys.max() + dy)

cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution_size=4)
cbar = fig.colorbar(hb, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Dead-zone density", fontsize=8)

from connroute.viz.style import save
save("coverage_heatmap", category="coverage", fig=fig)