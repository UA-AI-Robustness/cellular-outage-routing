# notebooks/15_timing_compare.py
import osmnx as ox, random, time
from tqdm import tqdm
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.tiles import build_tiles
from connroute.search.tile_heuristic import build_tile_heuristic
from connroute.search.lexico import constrained_route
from connroute.search.mats import constrained_mats

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
B=100.0

def time_algo(fn):
    t0=time.perf_counter()
    exp=0
    for (s,dst) in pairs:
        r=fn(s,dst)
        exp += r
    return (time.perf_counter()-t0)*1000, exp   # total ms, total expansions

# 1) old constrained (no expansion count available -> 0)
def old(s,dst):
    constrained_route(G,s,dst,budget_attr="d_dead",budget=B); return 0
t_old,_ = time_algo(old)
print(f"old constrained_route      : {t_old:8.1f} ms total ({t_old/len(pairs):.1f} ms/query)")

# 2) MATS no heuristic
def mats_noh(s,dst):
    _,_,_,e,_ = constrained_mats(G,s,dst,None if False else part, budget=B, use_heuristic=False); return e

# 3) MATS with heuristic, per tile size  (precompute timed separately)
for N in (16,32,64):
    part = build_tiles(G,N)   # tile build (one-time, per graph)
    # time no-heur once (independent of N, but uses part only for signature)
    tnoh,enoh = time_algo(lambda s,dst: constrained_mats(G,s,dst,part,budget=B,use_heuristic=False)[3])
    th,eh     = time_algo(lambda s,dst: constrained_mats(G,s,dst,part,budget=B,use_heuristic=True)[3])
    print(f"[{N}x{N}] no-heur: {tnoh:7.1f} ms ({enoh:,} exp) | "
          f"heur: {th:7.1f} ms ({eh:,} exp) | "
          f"time red={100*(1-th/max(tnoh,1e-9)):4.1f}%  exp red={100*(1-eh/max(enoh,1)):4.1f}%")