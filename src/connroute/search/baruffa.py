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
import numpy as np

def _snr_lin(db):
    return 10.0 ** (db / 10.0)


def compute_radio_weight(G, kind="on_off", s_ref_db=25.0, gamma=1.0,
                         d_max=None, beta=0.2):
    """Attach R(e) in [0,1] to each edge under a kind-specific key
    'R_baruffa_<kind>'. 1 = best coverage.
    kind: 'on_off'    -> R = covered fraction = 1 - dead_fraction
          'amplitude' -> Baruffa Eq. 14: R = 1 / d^gamma  (normalized)
          'capacity'  -> Baruffa Eq. 15: R = 1 - log2(d)/log2(D_MAX)
          'tent'      -> Baruffa Eq. 16: R = (1 - d/D_MAX)^beta
    d_max: coverage radius D^(MAX) in metres for amplitude/capacity/tent;
           defaults to the tower search radius if not given.
    """
    if d_max is None:
        d_max = 2000.0

    if kind == "amplitude":
        raw = []
        for _, _, data in G.edges(data=True):
            d = max(float(data.get("d_mean_serve", d_max)), 1.0)
            raw.append(1.0 / (d ** gamma))
        raw = np.array(raw)
        ref = raw.max() if raw.max() > 0 else 1.0

    key = f"R_baruffa_{kind}"   # <-- kind-specific storage

    for _, _, data in G.edges(data=True):
        if kind == "on_off":
            R = 1.0 - float(data["dead_fraction"])
        elif kind == "amplitude":
            d = max(float(data.get("d_mean_serve", d_max)), 1.0)
            R = (1.0 / (d ** gamma)) / ref
        elif kind == "capacity":
            d = max(float(data.get("d_mean_serve", d_max)), 1.0)
            R = 1.0 - (math.log2(d) / math.log2(d_max))
        elif kind == "tent":
            d = float(data.get("d_mean_serve", d_max))
            R = max(1.0 - d / d_max, 0.0) ** beta
        else:
            raise ValueError(f"unknown radio weight kind: {kind}")
        data[key] = min(max(R, 0.0), 1.0)   # <-- store under key, not "R_baruffa"

def baruffa_route(G, s, d, alpha, kind="on_off", length_attr="length"):
    """Dijkstra on Baruffa's cumulative cost g (Eq. 22-23).
    Requires compute_radio_weight(G, kind=kind) to have set
    'R_baruffa_<kind>' on edges. Returns (path, g_cost) or ([], inf)."""
    key = f"R_baruffa_{kind}"
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
            R = float(ed[key])
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