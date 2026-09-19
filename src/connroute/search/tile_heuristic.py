"""Module 1 -- generalized tile heuristic (per-operator, admissible).

Builds a coarse tile graph from cross-edges and runs a backward search from
each destination tile to give an admissible remaining-cost bound per node:

  additive (cost, minimized): W_tile = min cross-edge cost; backward ADDITIVE
      Dijkstra -> lower bound on remaining cost (optimistic).
  bottleneck (rate, maximized): W_tile = max cross-edge rate; backward MAX-MIN
      -> upper bound on best reachable bottleneck (optimistic).

These feed the generalized search (Module 2) as h(node) = H[dest_tile][tile_of[node]].
"""
from __future__ import annotations
import heapq
from collections import defaultdict


def _tile_weights(part, edge_val, agg):
    """Aggregate cross-edge values into per-adjacency tile weights.
    edge_val: (u,v,k) -> float. agg: 'min' (additive) or 'max' (bottleneck)."""
    W = {}
    for (ta, tb), edges in part.cross_edges.items():
        vals = [edge_val(u, v, k) for (u, v, k) in edges]
        if not vals:
            continue
        W[(ta, tb)] = min(vals) if agg == "min" else max(vals)
    return W


def backward_additive(dest_tile, tiles, W):
    """Least additive tile-cost from every tile to dest_tile (lower bound)."""
    preds = defaultdict(list)
    for (ta, tb), w in W.items():
        preds[tb].append((ta, w))          # edge ta->tb, so tb's predecessor is ta
    INF = float("inf")
    H = {t: INF for t in tiles}
    H[dest_tile] = 0.0
    pq = [(0.0, dest_tile)]
    while pq:
        c, ti = heapq.heappop(pq)
        if c > H[ti]:
            continue
        for ta, w in preds.get(ti, []):
            nc = c + w
            if nc < H[ta]:
                H[ta] = nc
                heapq.heappush(pq, (nc, ta))
    return H


def backward_maxmin(dest_tile, tiles, W):
    """Best reachable bottleneck (max over paths of min edge) to dest_tile."""
    preds = defaultdict(list)
    for (ta, tb), w in W.items():
        preds[tb].append((ta, w))
    H = {t: 0.0 for t in tiles}
    H[dest_tile] = float("inf")            # identity for min-aggregation
    pq = [(-float("inf"), dest_tile)]
    while pq:
        neg, ti = heapq.heappop(pq)
        s = -neg
        if s < H[ti]:
            continue
        for ta, w in preds.get(ti, []):
            cand = min(s, w)               # bottleneck: worst of path-so-far and this edge
            if cand > H[ta]:
                H[ta] = cand
                heapq.heappush(pq, (-cand, ta))
    return H


def build_tile_heuristic(part, dest_tile, edge_val, operator):
    """Return H: tile -> admissible remaining bound toward dest_tile.
    operator: 'additive' or 'bottleneck'."""
    tiles = list(part.tiles.keys())
    if operator == "additive":
        W = _tile_weights(part, edge_val, agg="min")
        return backward_additive(dest_tile, tiles, W)
    elif operator == "bottleneck":
        W = _tile_weights(part, edge_val, agg="max")
        return backward_maxmin(dest_tile, tiles, W)
    else:
        raise ValueError(f"unknown operator: {operator}")