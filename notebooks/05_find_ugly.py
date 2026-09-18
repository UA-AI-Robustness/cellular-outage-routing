# notebooks/05_find_ugly.py
import osmnx as ox, numpy as np, random
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","q","q_dwell","time","length"): d[a]=float(d[a])
for _,nd in G.nodes(data=True): nd["x"]=float(nd["x"]); nd["y"]=float(nd["y"])

def sl(s,d): return float(np.hypot(G.nodes[s]["x"]-G.nodes[d]["x"], G.nodes[s]["y"]-G.nodes[d]["y"]))
def dt(path):
    dd=tt=0.0
    for a,b in zip(path[:-1],path[1:]):
        ed=min(G[a][b].values(), key=lambda e:float(e["time"]))
        dd+=float(ed["d_dead"]); tt+=float(ed["time"])
    return dd,tt

prefs = build_preferences(cfg)
base = make_order(prefs,["time"]); conn = make_order(prefs,["dead_exposure","time"])
rng = random.Random(cfg.seed); nodes=list(G.nodes())

print("looking for pairs where connectivity route detours a LOT...")
found=0
for _ in range(3000):
    s,d = rng.choice(nodes), rng.choice(nodes)
    if s==d or sl(s,d)<2500: continue
    fp,_ = lexico_route(G,s,d,base)
    if not fp: continue
    fd,ft = dt(fp)
    if fd < 100: continue                     # fast route must cross a real hole
    cp,_ = lexico_route(G,s,d,conn, eps=(50.0,0.0))
    if not cp: continue
    cd,ct = dt(cp)
    detour = 100*(ct-ft)/ft
    if detour > 40:                           # a "bad" detour
        found+=1
        print(f"\nUGLY pair {s}->{d}:")
        print(f"  fastest         : dead={fd:7.0f}m time={ft:6.0f}s nodes={len(fp)}")
        print(f"  connectivity    : dead={cd:7.0f}m time={ct:6.0f}s nodes={len(cp)}  detour={detour:.0f}%")
        if found>=3: break
print("\ndone")