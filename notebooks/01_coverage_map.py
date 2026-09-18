"""Coverage map in house style: green=covered -> red=dead."""
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from connroute.config import load_config
from connroute.signal.layer import cache_path
from connroute.viz.style import apply_style, GREEN, RED, save

apply_style(usetex=False)   # citywide map: skip usetex, override size below

cfg = load_config()
Gp = ox.load_graphml(cache_path(cfg))
dead = np.array([float(d["dead_fraction"]) for _, _, d in Gp.edges(data=True)])

cmap = LinearSegmentedColormap.from_list("cov", [GREEN, "#F4D03F", RED])
colors = [cmap(x) for x in dead]

fig, ax = ox.plot_graph(
    Gp, edge_color=colors, edge_linewidth=0.8,
    node_size=0, bgcolor="white", show=False, close=False,
    figsize=(7, 7),
)
save("coverage_map", category="coverage", fig=fig)   # -> results/figures/coverage/coverage_map_<ts>.pdf + .png