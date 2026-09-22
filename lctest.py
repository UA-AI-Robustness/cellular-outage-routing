import time, osmnx as ox
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from experiments import workers
cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","d_lo_model","d_lo_traf","time","length"):
        if a in d: d[a]=float(d[a])
for _,nd in G.nodes(data=True): nd["x"]=float(nd["x"]); nd["y"]=float(nd["y"])
nodes=list(G.nodes())
s,dst = nodes[0], nodes[400]
t0=time.time()
rows = workers.loadcompare_pair(G, cfg, (s,dst))
print(f"one pair took {time.time()-t0:.1f}s, {len(rows)} rows")
