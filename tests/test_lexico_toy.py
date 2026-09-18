"""Correctness test: brute-force every simple path on a tiny graph and check
lexico_route returns the true lexicographic optimum. The critical case is a
bottleneck (min) preference, where single-label search may be wrong.
"""
import itertools
import math
import networkx as nx

from connroute.search.preferences import Preference
from connroute.search.lexico import lexico_route, lex_greater


# ---- tiny preference builders over explicit edge attrs (no config needed) ----
_SUM_ID, _MIN_ID, _WORST = 0.0, math.inf, -math.inf

def pref_sum_min(attr):   # additive, minimized -> store -attr, maximize
    return Preference(attr, "sum", _SUM_ID, _WORST, lambda e: -float(e[attr]), "min")

def pref_min_max(attr):   # bottleneck, maximized
    return Preference(attr, "min", _MIN_ID, _WORST, lambda e: float(e[attr]), "max")


def build_toy():
    """A small graph with several s->d paths and deliberately tricky bottlenecks.
    Edge attrs: 'time' (minimize), 'q' (bottleneck-maximize)."""
    G = nx.MultiDiGraph()
    E = [
        # u, v, time, q
        (0, 1, 1.0, 0.9),
        (1, 5, 1.0, 0.2),   # fast path 0-1-5: short time, but a low-q bottleneck (0.2)
        (0, 2, 1.0, 0.5),
        (2, 3, 1.0, 0.5),
        (3, 5, 1.0, 0.5),   # longer path 0-2-3-5: more time, higher min-q (0.5)
        (0, 4, 2.0, 0.8),
        (4, 5, 2.0, 0.8),   # longest 0-4-5: most time, highest min-q (0.8)
    ]
    for u, v, t, q in E:
        G.add_edge(u, v, time=t, q=q)
    return G


def brute_force_optimum(G, s, d, order):
    """Enumerate all simple paths, compute each g-vector, return the lex-best."""
    best_g, best_path = None, None
    for path in nx.all_simple_paths(G, s, d):
        g = tuple(p.identity for p in order)
        for a, b in zip(path[:-1], path[1:]):
            edata = min(G[a][b].values(), key=lambda e: 0)  # single edge here
            g = tuple(p.combine(g[i], p.contrib(edata)) for i, p in enumerate(order))
        if best_g is None or lex_greater(g, best_g):
            best_g, best_path = g, path
    return best_path, best_g


def check(order, label):
    G = build_toy()
    s, d = 0, 5
    bf_path, bf_g = brute_force_optimum(G, s, d, order)
    alg_path, alg_g = lexico_route(G, s, d, order)
    ok = (alg_g == bf_g)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    print(f"    brute-force: path={bf_path}  g={tuple(round(x,3) for x in bf_g)}")
    print(f"    lexico     : path={alg_path}  g={tuple(round(x,3) for x in alg_g)}")
    return ok


if __name__ == "__main__":
    results = []
    # additive-only: should pass (single-label is provably correct here)
    results.append(check([pref_sum_min("time")], "time only (additive)"))
    # THE CRITICAL ONE: bottleneck first, then time
    results.append(check([pref_min_max("q"), pref_sum_min("time")], "live_call (min) > time  [CRITICAL]"))
    # reverse order
    results.append(check([pref_sum_min("time"), pref_min_max("q")], "time > live_call (min)"))

    print("\n" + ("ALL PASS" if all(results) else "SOME FAILED — engine needs the bottleneck fix"))