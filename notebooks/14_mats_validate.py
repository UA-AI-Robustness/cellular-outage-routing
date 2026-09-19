# notebooks/14_mats_validate.py
import osmnx as ox, random
from tqdm import tqdm
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.tiles import build_tiles
from connroute.search.lexico import constrained_route      # ground truth (no heuristic)
from connroute.search.mats import constrained_mats         # new guided search

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","q","d_lowrate","time","length"):
        if a in d: d[a]=float(d[a])
for _,nd in G.nodes(data=True): nd["x"]=float(nd["x"]); nd["y"]=float(nd["y"])

rng = random.Random(cfg.seed); nodes=list(G.nodes())
pairs=[]
while len(pairs)<40:
    s,dst=rng.choice(nodes),rng.choice(nodes)
    if s!=dst: pairs.append((s,dst))

B = 100.0
for N in (8, 16, 32,64):
    part = build_tiles(G, N)
    mismatches=0; exp_noh=0; exp_h=0; n=0
    for (s,dst) in tqdm(pairs, desc=f"{N}x{N}", unit="pair"):
        # ground truth: existing constrained (no heuristic)
        p0,c0,_ = constrained_route(G,s,dst, budget_attr="d_dead", budget=B)
        # new: no-heuristic (expansions baseline) and heuristic-guided
        _,c1,_,e1,_ = constrained_mats(G,s,dst,part, budget=B, use_heuristic=False)
        _,c2,_,e2,_ = constrained_mats(G,s,dst,part, budget=B, use_heuristic=True)
        n+=1; exp_noh+=e1; exp_h+=e2
        # costs must all match (same optimal)
        cs=[c for c in (c0,c1,c2) if c is not None]
        if cs and (max(cs)-min(cs) > 1e-6): mismatches+=1
    print(f"\n[{N}x{N}] mismatches={mismatches}/{n}  "
          f"expansions: no-heur={exp_noh:,}  heur={exp_h:,}  "
          f"reduction={100*(1-exp_h/max(exp_noh,1)):.1f}%")