# notebooks/24_kshortest_smoke.py
import osmnx as ox, random
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.kshortest import kshortest_rerank
from connroute.search.lexico import lexico_route, constrained_route
from connroute.search.preferences import build_preferences, make_order

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","time","length"):
        if a in d: d[a]=float(d[a])
for _,nd in G.nodes(data=True): nd["x"]=float(nd["x"]); nd["y"]=float(nd["y"])

prefs=build_preferences(cfg); rng=random.Random(cfg.seed); nodes=list(G.nodes())
def dt(path):
    dd=tt=0.0
    for a,b in zip(path[:-1],path[1:]):
        ed=min(G[a][b].values(), key=lambda e:float(e["time"]))
        dd+=float(ed["d_dead"]); tt+=float(ed["time"])
    return dd,tt

s=dd=None
for _ in range(3000):
    a,b=rng.choice(nodes),rng.choice(nodes)
    if a==b: continue
    fp,_=lexico_route(G,a,b,make_order(prefs,["time"]))
    if fp and dt(fp)[0]>400: s,dd,fast=a,b,fp; break

fdead,ftime=dt(fast)
print(f"pair {s}->{dd}")
print(f"fastest:          dead={fdead:5.0f}m time={ftime:5.0f}s")
for k in (3,5,10,20):
    kp,kexp,kt=kshortest_rerank(G,s,dd,k=k)
    det=100*(kt-ftime)/ftime if ftime else 0
    print(f"k-shortest k={k:2d}: dead={kexp:5.0f}m time={kt:5.0f}s  detour={det:+.0f}%")
cp,_,_=constrained_route(G,s,dd,budget_attr="d_dead",budget=100.0)
cdead,ctime=dt(cp)
print(f"ours (B=100):     dead={cdead:5.0f}m time={ctime:5.0f}s  detour={100*(ctime-ftime)/ftime:+.0f}%")