# notebooks/23_baruffa_smoke.py
import osmnx as ox, random
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.baruffa import compute_radio_weight, baruffa_route
from connroute.search.lexico import lexico_route
from connroute.search.preferences import build_preferences, make_order

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","dead_fraction","time","length","s_mean"):
        if a in d: d[a]=float(d[a])
for _,nd in G.nodes(data=True): nd["x"]=float(nd["x"]); nd["y"]=float(nd["y"])

def dead_time(path):
    dd=tt=0.0
    for a,b in zip(path[:-1],path[1:]):
        ed=min(G[a][b].values(), key=lambda e:float(e["time"]))
        dd+=float(ed["d_dead"]); tt+=float(ed["time"])
    return dd,tt

compute_radio_weight(G, kind="on_off")
prefs=build_preferences(cfg); rng=random.Random(cfg.seed); nodes=list(G.nodes())
# a pair whose fast route crosses a hole
s=dd=None
for _ in range(3000):
    a,b=rng.choice(nodes),rng.choice(nodes)
    if a==b: continue
    fp,_=lexico_route(G,a,b,make_order(prefs,["time"]))
    if fp and dead_time(fp)[0]>400: s,dd=a,b; fast=fp; break

print(f"pair {s}->{dd}")
fdead,ftime=dead_time(fast)
print(f"fastest:        dead={fdead:5.0f}m time={ftime:5.0f}s")
for alpha in (0.0, 0.5, 1.0, 2.0, 5.0):
    bp,_=baruffa_route(G,s,dd,alpha)
    if bp:
        bdead,btime=dead_time(bp)
        det=100*(btime-ftime)/ftime if ftime else 0
        print(f"baruffa a={alpha:4.1f}: dead={bdead:5.0f}m time={btime:5.0f}s  detour={det:+.0f}%")