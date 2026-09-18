# notebooks/11_diagnose_upload.py
import osmnx as ox, time
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","q","q_dwell","time","length"): d[a]=float(d[a])

# one CLOSE pair so it can't run forever
nodes = list(G.nodes()); s, d = nodes[0], nodes[30]
prefs = build_preferences(cfg)
fp,_ = lexico_route(G, s, d, make_order(prefs,["time"]))
print(f"pair {s}->{d}, fastest route has {len(fp)} nodes")

# instrument the upload search: count labels per node as it runs
import heapq, itertools
def diagnose(G, s, d, time_budget):
    counter = itertools.count()
    best = {s: [(0.0,0.0)]}
    pq = [(-0.0, 0.0, next(counter), s)]
    max_labels = 0; total_pushes = 0; t0 = time.time()
    while pq:
        if time.time()-t0 > 20:   # hard stop after 20s
            print(f"  STOPPED after 20s. max labels on any node: {max_labels}, pushes: {total_pushes:,}")
            return
        neg_data, t, _, u = heapq.heappop(pq)
        data = -neg_data
        if not any(abs(c-data)<1e-9 and abs(b-t)<1e-9 for (c,b) in best.get(u,[])):
            continue
        if u == d: 
            print(f"  DONE in {time.time()-t0:.2f}s. max labels: {max_labels}, pushes: {total_pushes:,}")
            return
        for _,v,k,ed in G.edges(u, keys=True, data=True):
            nt = t + float(ed["time"])
            if nt > time_budget: continue
            nd = data + float(ed["q_dwell"])
            labels = best.setdefault(v, [])
            if any(c>=nd-1e-9 and b<=nt+1e-9 for (c,b) in labels): continue
            labels[:] = [(c,b) for (c,b) in labels if not (nd>=c-1e-9 and nt<=b-1e-9)]
            labels.append((nd,nt))
            max_labels = max(max_labels, len(labels))
            total_pushes += 1
            heapq.heappush(pq, (-nd, nt, next(counter), v))

# fastest time for this pair
ft = sum(min(float(e["time"]) for e in G[a][b].values()) for a,b in zip(fp[:-1],fp[1:]))
print(f"fastest time = {ft:.0f}s")
for detour in (0.0, 0.10, 0.40):
    print(f"detour budget +{int(detour*100)}%:")
    diagnose(G, s, d, ft*(1+detour))