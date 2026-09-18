# notebooks/06_test_constrained.py
import osmnx as ox
from tqdm import tqdm
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route, constrained_route

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","q","q_dwell","time","length"): d[a]=float(d[a])

prefs = build_preferences(cfg); base = make_order(prefs,["time"])
def dt(path):
    dd=tt=0.0
    for a,b in zip(path[:-1],path[1:]):
        ed=min(G[a][b].values(), key=lambda e:float(e["time"]))
        dd+=float(ed["d_dead"]); tt+=float(ed["time"])
    return dd,tt

pairs = [(65302199,65306583),(65289075,2317635493),(65314378,65356292)]
for (s,d) in tqdm(pairs, desc="pairs", unit="pair"):
    fp,_ = lexico_route(G,s,d,base); fd,ft = dt(fp)
    tqdm.write(f"\n{s}->{d}")
    tqdm.write(f"  fastest        : dead={fd:6.0f}m time={ft:5.0f}s")
    for B in (50, 100, 200, 400):
        cp,ct,cb = constrained_route(G,s,d, budget=float(B))
        if not cp:
            tqdm.write(f"  budget={B:4d}m   : INFEASIBLE"); continue
        det = 100*(ct-ft)/ft
        tqdm.write(f"  budget={B:4d}m   : dead={cb:6.0f}m time={ct:5.0f}s detour={det:4.0f}%  nodes={len(cp)}")