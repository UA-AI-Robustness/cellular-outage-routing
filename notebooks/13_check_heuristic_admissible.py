# notebooks/13_check_heuristic_admissible.py
import osmnx as ox, random
from tqdm import tqdm
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.tiles import build_tiles
from connroute.search.tile_heuristic import build_tile_heuristic
from connroute.search.lexico import constrained_route  # gives true optimal cost

cfg = load_config(); G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
for _,_,d in G.edges(data=True):
    for a in ("d_dead","q","d_lowrate","time","length"):
        if a in d:
            d[a] = float(d[a])
for _,nd in G.nodes(data=True): nd["x"]=float(nd["x"]); nd["y"]=float(nd["y"])

part = build_tiles(G, 16)

# ADDITIVE test: heuristic on 'time' must be <= true shortest time to dest
def edge_time(u,v,k): return float(G[u][v][k]["time"])

rng = random.Random(cfg.seed); nodes=list(G.nodes())
violations = 0; checked = 0
for _ in tqdm(range(30), desc="admissibility"):
    d = rng.choice(nodes)
    H = build_tile_heuristic(part, part.tile_of[d], edge_time, "additive")
    # true shortest time from a few sources to d (Dijkstra on time)
    import networkx as nx
    lengths = nx.single_source_dijkstra_path_length(G.reverse(copy=False), d, weight="time")
    for s in rng.sample(nodes, 20):
        if s not in lengths: continue
        h = H.get(part.tile_of[s], 0.0)
        true_rem = lengths[s]
        checked += 1
        if h > true_rem + 1e-6:        # ADMISSIBILITY VIOLATION: h overestimates cost
            violations += 1

print(f"\nadditive-time heuristic: {violations}/{checked} admissibility violations")
print("PASS -- admissible" if violations == 0 else "FAIL -- heuristic overestimates, not admissible")