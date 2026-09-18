import osmnx as ox, numpy as np
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","q","q_dwell","time","length"): d[a]=float(d[a])

s, d = 65313342, 65315947   # <-- REPLACE with example 2's s->d from your terminal

prefs = build_preferences(cfg)
def dead_time(path):
    dd=tt=0.0
    for a,b in zip(path[:-1],path[1:]):
        ed=min(G[a][b].values(), key=lambda e:float(e["time"]))
        dd+=float(ed["d_dead"]); tt+=float(ed["time"])
    return dd,tt

fast,_ = lexico_route(G,s,d, make_order(prefs,["time"]))
conn,_ = lexico_route(G,s,d, make_order(prefs,["dead_exposure","time"]), eps=(50.0,0.0))
fd,ft = dead_time(fast); cd,ct = dead_time(conn)
print(f"fastest          : dead={fd:7.1f}m  time={ft:7.1f}s  nodes={len(fast)}")
print(f"connectivity(e50): dead={cd:7.1f}m  time={ct:7.1f}s  nodes={len(conn)}")