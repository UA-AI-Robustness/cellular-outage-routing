# notebooks/25_traffic_test.py
import osmnx as ox
from connroute.config import load_config
from connroute.graph.build import cache_path as graph_cache_path
from connroute.signal.traffic import build_traffic_grid
import numpy as np

cfg = load_config()
G = ox.load_graphml(graph_cache_path(cfg))
grid = build_traffic_grid(G, cfg, grid_side=12)
rho = np.array(grid["rho"])
print(f"congestion grid: min={rho.min():.2f}  mean={rho.mean():.2f}  max={rho.max():.2f}")
print(f"congested cells (rho<0.7): {(rho<0.7).sum()}/{rho.size}")