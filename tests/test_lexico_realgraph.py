import pytest
pytest.skip("heavy: brute-force enumeration exhausts memory on this graph; "
            "run only on a high-memory machine. Primary correctness is "
            "test_lexico_toy.py (full exhaustive enumeration on small cases).",
            allow_module_level=True)

"""Brute-force validation on SMALL INDUCED SUBGRAPHS of the real SF graph.
Engine and brute-force run on the same subgraph, so they search the identical
path space -> a fair, memory-safe correctness check."""
import random, itertools
import networkx as nx
import osmnx as ox

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route, lex_greater

RADIUS_HOPS = 2      # subgraph = nodes within this many hops of s
N_PAIRS = 10
MAX_PATHS = 300     # skip a pair if its subgraph has more paths than this


def load_graph():
    cfg = load_config()
    G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
    for _, _, dta in G.edges(data=True):
        for a in ("d_dead", "q", "q_dwell", "time", "length"):
            dta[a] = float(dta[a])
    return G, cfg


def subgraph_around(G, s, radius):
    """Induced subgraph of nodes within `radius` hops of s (undirected reach)."""
    reach = nx.single_source_shortest_path_length(G, s, cutoff=radius)
    return G.subgraph(reach.keys()).copy()


def brute_force(H, s, d, order):
    best = None; n = 0
    for node_path in nx.all_simple_paths(H, s, d):
        key_choices = [[(a, b, k) for k in H[a][b]] for a, b in zip(node_path[:-1], node_path[1:])]
        for combo in itertools.product(*key_choices):
            n += 1
            if n > MAX_PATHS:
                return None, n     # too big -> signal skip
            g = tuple(p.identity for p in order)
            for (a, b, k) in combo:
                ed = H[a][b][k]
                g = tuple(p.combine(g[i], p.contrib(ed)) for i, p in enumerate(order))
            if best is None or lex_greater(g, best):
                best = g
    return best, n


if __name__ == "__main__":
    G, cfg = load_graph()
    prefs = build_preferences(cfg)
    orders = {
        "time": make_order(prefs, ["time"]),
        "dead>time": make_order(prefs, ["dead_exposure", "time"]),
        "live_call>time": make_order(prefs, ["live_call", "time"]),
        "upload>time": make_order(prefs, ["upload", "time"]),
    }
    rng = random.Random(cfg.seed)
    nodes = list(G.nodes())

    total = passed = tested_pairs = 0
    while tested_pairs < N_PAIRS:
        s = rng.choice(nodes)
        H = subgraph_around(G, s, RADIUS_HOPS)
        cand = [n for n in H.nodes() if n != s and nx.has_path(H, s, n)]
        if not cand:
            continue
        d = rng.choice(cand)
        # quick skip if subgraph is too path-rich
        bf0, n0 = brute_force(H, s, d, orders["time"])
        if bf0 is None:
            continue
        tested_pairs += 1
        for label, order in orders.items():
            bf, _ = brute_force(H, s, d, order)
            _, alg = lexico_route(H, s, d, order)   # ENGINE ON SAME SUBGRAPH
            total += 1
            ok = bf and alg and all(abs(a-b) < 1e-6 for a, b in zip(alg, bf))
            if ok:
                passed += 1
            else:
                print(f"[FAIL] {s}->{d} {label}: brute={tuple(round(x,3) for x in bf) if bf else None}  algo={tuple(round(x,3) for x in alg) if alg else None}")

    print(f"\n{passed}/{total} checks passed  ({tested_pairs} pairs)")
    print("ALL PASS" if passed == total else "SOME FAILED — real engine bug, investigate")