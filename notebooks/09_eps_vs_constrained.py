# notebooks/09_eps_vs_constrained.py
"""Decide the algorithm: can eps-tuned lexico_route match constrained_route?
For the known ugly pairs, compare constrained(B) vs lexico([dead,time], eps)
across a sweep of eps, on the two things that matter: dead removed and detour."""
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
conn_order = make_order(prefs,["dead_exposure","time"])

def dt(path):
    dd=tt=0.0
    for a,b in zip(path[:-1],path[1:]):
        ed=min(G[a][b].values(), key=lambda e:float(e["time"]))
        dd+=float(ed["d_dead"]); tt+=float(ed["time"])
    return dd,tt

pairs = [(65302199,65306583),(65289075,2317635493),(65314378,65356292)]
EPS = [0, 25, 50, 100, 200, 400]     # dead-exposure tolerance (metres)

for (s,d) in tqdm(pairs, desc="pairs", unit="pair"):
    fp,_ = lexico_route(G,s,d,base); fd,ft = dt(fp)
    tqdm.write(f"\n{s}->{d}   fastest: dead={fd:.0f}m time={ft:.0f}s")
    tqdm.write("  --- constrained (min time s.t. dead<=B) ---")
    for B in (50,100,200):
        cp,_,_ = constrained_route(G,s,d,budget=float(B))
        if cp:
            cd,ct = dt(cp); tqdm.write(f"    B={B:4d}m : dead={cd:5.0f}m time={ct:5.0f}s detour={100*(ct-ft)/ft:4.0f}%")
    tqdm.write("  --- lexico [dead,time] with eps ---")
    for e in EPS:
        lp,_ = lexico_route(G,s,d,conn_order, eps=(float(e),0.0))
        if lp:
            ld,lt = dt(lp); tqdm.write(f"    eps={e:4d}m: dead={ld:5.0f}m time={lt:5.0f}s detour={100*(lt-ft)/ft:4.0f}%")