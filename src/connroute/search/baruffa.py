"""Baruffa et al. (IEEE Access 2026) radio-coverage-aware path planning,
reproduced as a routing baseline. Per-edge scalarized cost (their Eq. 22-23):

    g(pi) = sum_e  max{1 - alpha * R(e), 0} * length(e)

where R(e) in [0,1] is a normalized radio weight (1 = best coverage) and
alpha in [0, inf) trades traveled distance against radio-coverage experience.
We provide their on-off (Eq. 13) and capacity-related (Eq. 15) radio weights,
mapped to our per-edge coverage estimate. Solved with Dijkstra on g.
"""
from __future__ import annotations
import heapq
import itertools
import math


def _snr_lin(db):
    return 10.0 ** (db / 10.0)


def compute_radio_weight(G, kind="on_off", s_ref_db=25.0):
    """Attach R(e) in [0,1] to each edge. 1 = best coverage.
    kind: 'on_off'  -> R = covered fraction = 1 - dead_fraction
          'capacity' -> Shannon-like, normalized to s_ref_db."""
    ref = math.log2(1.0 + _snr_lin(s_ref_db))
    for _, _, data in G.edges(data=True):
        if kind == "on_off":
            R = 1.0 - float(data["dead_fraction"])
        elif kind == "capacity":
            s = float(data.get("s_mean", -math.inf))
            if not math.isfinite(s):
                R = 0.0
            else:
                R = math.log2(1.0 + _snr_lin(s)) / ref
        else:
            raise ValueError(f"unknown radio weight kind: {kind}")
        data["R_baruffa"] = min(max(R, 0.0), 1.0)   # clip to [0,1]


def baruffa_route(G, s, d, alpha, length_attr="length"):
    """Dijkstra on Baruffa's cumulative cost g (Eq. 22-23).
    Requires compute_radio_weight(G, ...) to have set 'R_baruffa' on edges.
    Returns (path, g_cost) or ([], inf)."""
    counter = itertools.count()
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, next(counter), s)]
    while pq:
        g, _, u = heapq.heappop(pq)
        if g > dist.get(u, math.inf):
            continue
        if u == d:
            break
        for _, v, k, ed in G.edges(u, keys=True, data=True):
            L = float(ed[length_attr])
            R = float(ed["R_baruffa"])
            edge_cost = max(1.0 - alpha * R, 0.0) * L      # Eq. 23
            ng = g + edge_cost
            if ng < dist.get(v, math.inf):
                dist[v] = ng
                prev[v] = u
                heapq.heappush(pq, (ng, next(counter), v))
    if d not in prev and s != d:
        return [], math.inf
    path = [d]
    while path[-1] != s:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist.get(d, math.inf)