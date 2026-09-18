# notebooks/03b_upload.py
import time, osmnx as ox
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route

cfg = load_config()
G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _, _, dta in G.edges(data=True):
    for a in ("d_dead","q","q_dwell","time","length"): dta[a]=float(dta[a])
nodes = list(G.nodes()); s, d = nodes[0], nodes[20]
prefs = build_preferences(cfg)

# time-first (upload as tiebreaker) should terminate instantly:
order = make_order(prefs, ["time", "upload"])
t0=time.time(); path,g = lexico_route(G,s,d,order)
print(f"[time, upload]: {time.time()-t0:.3f}s  {len(path)} nodes")