"""Smoke test: run one lexicographic route on the SF objectives graph."""
import osmnx as ox
from connroute.config import load_config
from connroute.objectives.attach import cache_path
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route, readable_scores

cfg = load_config()
G = ox.load_graphml(cache_path(cfg))
# GraphML -> floats for the attrs the search reads
from connroute.objectives.attach import _coerce_types
_coerce_types(G)
for _, _, dta in G.edges(data=True):
    for a in ("d_dead", "q", "q_dwell", "time"):
        dta[a] = float(dta[a])

nodes = list(G.nodes())
s, d = nodes[0], nodes[len(nodes)//2]   # arbitrary OD pair

prefs = build_preferences(cfg)

for order_names in (["time"], ["dead_exposure", "time"], ["live_call", "time"]):
    order = make_order(prefs, order_names)
    path, g = lexico_route(G, s, d, order)
    if not path:
        print(f"{order_names}: no path"); continue
    sc = readable_scores(g, order)
    print(f"{str(order_names):35s} -> {len(path)} nodes  scores={ {k: round(v,2) for k,v in sc.items()} }")