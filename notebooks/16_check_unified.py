# notebooks/16_check_unified.py
"""Check the unification: bound-mode == constrained_route, optimize-mode ==
lexico_route, and demonstrate the motivating-failure contrast on a pair whose
fastest route crosses a dead zone."""
import osmnx as ox, random
from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.lexico import constrained_route, lexico_route
from connroute.search.preferences import build_preferences, make_order
from connroute.search.unified import unified_route

cfg = load_config()
G = ox.load_graphml(cache_path(cfg))
_coerce_types(G)
for _, _, d in G.edges(data=True):
    for a in ("d_dead", "q", "d_lowrate", "time", "length"):
        if a in d:
            d[a] = float(d[a])
for _, nd in G.nodes(data=True):
    nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])

rng = random.Random(cfg.seed)
nodes = list(G.nodes())


def dt(path):
    """Sum dead-exposure and travel time along a node path."""
    dd = tt = 0.0
    for a, b in zip(path[:-1], path[1:]):
        ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
        dd += float(ed["d_dead"]); tt += float(ed["time"])
    return dd, tt


# --- find a pair whose FASTEST route crosses a dead zone (so the contrast shows) ---
prefs = build_preferences(cfg)
fast_order = make_order(prefs, ["time"])
s = d = None
for _ in range(8000):
    cs, cd = rng.choice(nodes), rng.choice(nodes)
    if cs == cd:
        continue
    fp, _ = lexico_route(G, cs, cd, fast_order)
    if not fp:
        continue
    fd, ft = dt(fp)
    if fd < 300:
        continue
    # test the actual contrast on this pair
    pb, _, _ = constrained_route(G, cs, cd, budget_attr="d_dead", budget=100.0)
    po, _ = lexico_route(G, cs, cd, make_order(prefs, ["dead_exposure", "time"]))
    if not pb or not po:
        continue
    _, tb = dt(pb)
    _, to = dt(po)
    if to > tb * 1.3:          # optimize is >30% slower than bound -> real contrast
        s, d = cs, cd
        fd_final = fd
        break

if s is None:
    raise SystemExit("no contrasting pair found")
print(f"pair {s}->{d}, fastest-route dead={fd_final:.0f}m\n")

# --- 1) bound mode == constrained_route (identical by construction) ---
p_u, _ = unified_route(G, s, d, cfg, objective="dead_exposure",
                       mode="bound", budget=100.0)
p_c, _, _ = constrained_route(G, s, d, budget_attr="d_dead", budget=100.0)
print("bound mode == constrained_route:", p_u == p_c)

# --- 2) optimize mode == lexico_route (identical, shows runaway detour) ---
p_o, _ = unified_route(G, s, d, cfg, objective="dead_exposure", mode="optimize")
p_l, _ = lexico_route(G, s, d, make_order(prefs, ["dead_exposure", "time"]))
print("optimize mode == lexico_route :", p_o == p_l)

# --- 3) the contrast: bound is sensible, optimize is a runaway detour ---
if p_u and p_o:
    du, tu = dt(p_u)
    do, to = dt(p_o)
    print(f"\nbound   : dead={du:5.0f}m time={tu:5.0f}s")
    print(f"optimize: dead={do:5.0f}m time={to:5.0f}s   <- the 'motivating failure'")
    if tu > 0:
        print(f"\noptimize detour vs bound: {100*(to-tu)/tu:+.0f}% time "
              f"to remove {du-do:.0f}m more dead")