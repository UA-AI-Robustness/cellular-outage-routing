# notebooks/12_livecall_divergence.py
import osmnx as ox, numpy as np, random
from tqdm import tqdm
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.lexico import constrained_route

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","d_lowrate","q","time","length"): d[a]=float(d[a])
for _,nd in G.nodes(data=True): nd["x"]=float(nd["x"]); nd["y"]=float(nd["y"])

# how many edges are "covered but slow" -- the edges that make live-call DISTINCT
covered_but_slow = sum(1 for _,_,d in G.edges(data=True)
                       if d["d_dead"]==0 and d["d_lowrate"]>0)
total = G.number_of_edges()
print(f"covered-but-slow edges: {covered_but_slow:,} / {total:,} ({100*covered_but_slow/total:.1f}%)")
print("  (if ~0%, live-call collapses into continuity; if meaningful, it's distinct)\n")

rng = random.Random(cfg.seed); nodes=list(G.nodes())
def sl(s,d): return np.hypot(G.nodes[s]["x"]-G.nodes[d]["x"], G.nodes[s]["y"]-G.nodes[d]["y"])

pairs=[]
while len(pairs)<200:
    s,d=rng.choice(nodes),rng.choice(nodes)
    if s!=d and sl(s,d)>=2000: pairs.append((s,d))

B = 100.0
differ = 0; n = 0
for (s,d) in tqdm(pairs, desc="pairs", unit="pair"):
    cont,_,_ = constrained_route(G,s,d, budget_attr="d_dead",    budget=B)
    call,_,_ = constrained_route(G,s,d, budget_attr="d_lowrate", budget=B)
    if not cont or not call: continue
    n += 1
    if cont != call: differ += 1

print(f"\ncontinuity vs live-call routes differ on {differ}/{n} pairs ({100*differ/max(n,1):.0f}%)")