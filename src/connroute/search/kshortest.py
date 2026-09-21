"""k-shortest-paths reranking baseline: generate the k fastest simple routes,
then pick the one with the least predicted below-threshold exposure. This is a
practical navigation-style heuristic (rerank a nav app's top-k alternatives),
and a fair baseline for 'why not just rerank the fastest few routes?'.
"""
from __future__ import annotations
import networkx as nx
from itertools import islice


def _simple_digraph(G, weight_attr="time"):
    """Collapse the MultiDiGraph to a simple DiGraph keeping, per (u,v), the
    min-weight parallel edge. networkx k-shortest works on simple graphs."""
    H = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        w = float(data[weight_attr])
        if H.has_edge(u, v):
            if w < H[u][v]["w"]:
                H[u][v]["w"] = w
        else:
            H.add_edge(u, v, w=w)
    return H


def kshortest_rerank(G, s, d, k=5, weight_attr="time", exposure_attr="d_dead",
                     _cache={}):
    """Return the least-exposure route among the k fastest simple routes.
    Returns (path, exposure, time) or ([], None, None)."""
    # cache the simple graph per weight_attr (built once, reused across queries)
    key = (id(G), weight_attr)
    H = _cache.get(key)
    if H is None:
        H = _simple_digraph(G, weight_attr)
        _cache[key] = H
    if s not in H or d not in H:
        return [], None, None

    def path_exposure_time(path):
        exp = tt = 0.0
        for a, b in zip(path[:-1], path[1:]):
            ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
            exp += float(ed[exposure_attr]); tt += float(ed["time"])
        return exp, tt

    try:
        gen = nx.shortest_simple_paths(H, s, d, weight="w")  # yields fastest-first
        candidates = list(islice(gen, k))
    except nx.NetworkXNoPath:
        return [], None, None
    if not candidates:
        return [], None, None

    best_path, best_exp, best_t = None, None, None
    for path in candidates:
        exp, tt = path_exposure_time(path)
        if best_exp is None or exp < best_exp:
            best_path, best_exp, best_t = path, exp, tt
    return best_path, best_exp, best_t