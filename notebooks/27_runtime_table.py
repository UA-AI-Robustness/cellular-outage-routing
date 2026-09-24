# notebooks/27_runtime_table.py
"""Query latency and search-effort table for the constrained search.
No changes to existing code — just times constrained_route calls directly."""
import time
import random
import numpy as np
import pandas as pd
import osmnx as ox

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.lexico import lexico_route, constrained_route
from connroute.search.preferences import build_preferences, make_order

cfg = load_config()
G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _, _, d in G.edges(data=True):
    for a in ("d_dead", "time", "length"):
        if a in d:
            d[a] = float(d[a])
for _, nd in G.nodes(data=True):
    nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])

prefs = build_preferences(cfg)
base = make_order(prefs, ["time"])
rng = random.Random(cfg.seed)
nodes = list(G.nodes())

def straight_line_m(s, d):
    return np.hypot(G.nodes[s]["x"]-G.nodes[d]["x"], G.nodes[s]["y"]-G.nodes[d]["y"])

pairs = []
while len(pairs) < 200:
    s, d = rng.choice(nodes), rng.choice(nodes)
    if s != d and straight_line_m(s, d) >= 2000.0:
        pairs.append((s, d))

BUDGETS = [100, 200, 400, 800, 1600]   # denser sweep, matches figures elsewhere
rows = []
for B in BUDGETS:
    for (s, d) in pairs:
        t0 = time.perf_counter()
        path, cost, bud_used = constrained_route(G, s, d, budget_attr="d_dead",
                                                  budget=float(B), cost_attr="time")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rows.append({"budget": B, "latency_ms": elapsed_ms,
                     "feasible": bool(path)})

df = pd.DataFrame(rows)
print(f"{'Budget':>8} {'n':>5} {'median ms':>10} {'p95 ms':>8} {'max ms':>8} {'feasible%':>10}")
for B in BUDGETS:
    sub = df[df["budget"] == B]
    print(f"{B:>7}m {len(sub):>5} {sub['latency_ms'].median():>10.2f} "
          f"{sub['latency_ms'].quantile(0.95):>8.2f} {sub['latency_ms'].max():>8.2f} "
          f"{100*sub['feasible'].mean():>9.1f}%")

df.to_csv("results/tables/runtime_latency.csv", index=False)
print("\nsaved results/tables/runtime_latency.csv")