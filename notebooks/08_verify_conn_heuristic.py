# notebooks/08_verify_conn_heuristic.py
import time, osmnx as ox, random
from tqdm import tqdm
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.lexico import constrained_route
from connroute.search.heuristics import constrained_route_h

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","q","q_dwell","time","length"): d[a]=float(d[a])
for _,nd in G.nodes(data=True): nd["x"]=float(nd["x"]); nd["y"]=float(nd["y"])

nodes = list(G.nodes()); rng = random.Random(cfg.seed)
pairs = []
while len(pairs) < 40:
    s,d = rng.choice(nodes), rng.choice(nodes)
    if s!=d: pairs.append((s,d))

mismatches=0; t_old=t_new=0.0
for (s,d) in tqdm(pairs, desc="verify", unit="pair"):
    for B in (50.0, 200.0):
        t0=time.time(); p1,c1,_ = constrained_route(G,s,d,budget=B);   t_old+=time.time()-t0
        t0=time.time(); p2,c2,_ = constrained_route_h(G,s,d,budget=B); t_new+=time.time()-t0
        if (c1 is None) != (c2 is None): mismatches+=1
        elif c1 is not None and abs(c1-c2) > 1e-6: mismatches+=1

print(f"\nmismatches: {mismatches} / {len(pairs)*2}")
print(f"old (plain)     total: {t_old:6.2f}s")
print(f"new (conn-heur) total: {t_new:6.2f}s   speedup: {t_old/max(t_new,1e-9):.1f}x")