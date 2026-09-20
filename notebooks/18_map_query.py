# notebooks/18_map_query.py
import osmnx as ox, matplotlib.pyplot as plt, contextily as cx, random
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.lexico import lexico_route, constrained_route
from connroute.search.preferences import build_preferences, make_order
from connroute.viz.style import apply_style, BLUE, save

apply_style(usetex=False)
cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","q","d_lowrate","time","length"):
        if a in d: d[a]=float(d[a])
for _,nd in G.nodes(data=True): nd["x"]=float(nd["x"]); nd["y"]=float(nd["y"])

prefs = build_preferences(cfg)
def dt(path):
    dd=tt=0.0
    for a,b in zip(path[:-1],path[1:]):
        ed=min(G[a][b].values(), key=lambda e:float(e["time"]))
        dd+=float(ed["d_dead"]); tt+=float(ed["time"])
    return dd,tt

# find a pair whose fast route crosses a dead zone and where constrained detours around it
rng=random.Random(cfg.seed); nodes=list(G.nodes())
s=d=None
for _ in range(5000):
    cs,cd=rng.choice(nodes),rng.choice(nodes)
    if cs==cd: continue
    fp,_=lexico_route(G,cs,cd,make_order(prefs,["time"]))
    if not fp: continue
    fd,_=dt(fp)
    if fd<400: continue
    cp,_,_=constrained_route(G,cs,cd,budget_attr="d_dead",budget=100.0)
    if cp and cp!=fp:
        s,d=cs,cd; fast=fp; conn=cp; break

fig, ax = plt.subplots(figsize=(6,7))
def xy(path): return [G.nodes[n]["x"] for n in path],[G.nodes[n]["y"] for n in path]
fx,fy=xy(fast); cx_,cy=xy(conn)
ax.plot(fx,fy,color=BLUE,lw=2.5,zorder=4,label="Fastest")
ax.plot(cx_,cy,color="#1a1a1a",lw=2.2,zorder=5,label="Connectivity-aware")
ax.scatter([G.nodes[s]["x"]],[G.nodes[s]["y"]],c="green",s=80,marker="o",zorder=6)
ax.scatter([G.nodes[d]["x"]],[G.nodes[d]["y"]],c="red",s=100,marker="*",zorder=6)
ax.set_axis_off()
cx.add_basemap(ax, crs="EPSG:32610", source=cx.providers.CartoDB.Positron, attribution_size=5)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
save("query_map", category="routes", fig=fig)
print(f"pair {s}->{d}")