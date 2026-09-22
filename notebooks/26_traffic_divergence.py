# notebooks/26_traffic_divergence.py
"""Does traffic-informed load change routing vs. modeled load?
Build rate-threshold call-inadequate distance from q (modeled) and q_tt
(traffic-informed), route on each, count divergence."""
import osmnx as ox, numpy as np, random
from tqdm import tqdm
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.lexico import constrained_route

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("q","q_tt","time","length"):
        if a in d: d[a]=float(d[a])
for _,nd in G.nodes(data=True): nd["x"]=float(nd["x"]); nd["y"]=float(nd["y"])

TAU = 0.30   # rate threshold (call/upload-grade); edges below this are "inadequate"

# build rate-inadequate distance from each load model
n_diff_edges = 0
for _,_,d in G.edges(data=True):
    L = float(d["length"])
    lo_model = 1.0 if d["q"]    < TAU else 0.0
    lo_traf  = 1.0 if d.get("q_tt", d["q"]) < TAU else 0.0
    d["d_lo_model"] = lo_model * L
    d["d_lo_traf"]  = lo_traf  * L
    if lo_model != lo_traf:
        n_diff_edges += 1

tot = G.number_of_edges()
print(f"edges whose adequacy FLIPS between load models: {n_diff_edges}/{tot} ({100*n_diff_edges/tot:.1f}%)")
print("  (if ~0%, traffic doesn't change adequacy -> routing won't differ)\n")

rng = random.Random(cfg.seed); nodes=list(G.nodes())
def sl(s,d): return np.hypot(G.nodes[s]["x"]-G.nodes[d]["x"], G.nodes[s]["y"]-G.nodes[d]["y"])
pairs=[]
while len(pairs)<300:
    s,d=rng.choice(nodes),rng.choice(nodes)
    if s!=d and sl(s,d)>=2000: pairs.append((s,d))

B=200.0; differ=0; n=0
for (s,d) in tqdm(pairs, desc="pairs", unit="pair"):
    rm,_,_ = constrained_route(G,s,d,budget_attr="d_lo_model", budget=B)
    rt,_,_ = constrained_route(G,s,d,budget_attr="d_lo_traf",  budget=B)
    if not rm or not rt: continue
    n+=1
    if rm != rt: differ+=1

print(f"\nmodeled-load vs traffic-load routes differ on {differ}/{n} pairs ({100*differ/max(n,1):.0f}%)")