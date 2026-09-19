"""Module 2 -- Mixed-Aggregation Tile-heuristic Search (constrained mode).

Minimizes travel time subject to a budget on an additive objective (e.g.
dead-exposure or call-inadequate distance), guided by the admissible additive
tile heuristic from Module 1. Returns the SAME optimal route as the
heuristic-free constrained search, with a node-expansion count so the
heuristic's effect can be measured.

Label = (time, budget_used) per node; dominance prunes; over-budget branches
are cut. Priority = time + h_time(node), where h_time is the admissible tile
lower-bound on remaining travel time.
"""
from __future__ import annotations
import heapq
import itertools
import time as _time

from connroute.search.tile_heuristic import build_tile_heuristic


def constrained_mats(G, s, d, part,
                     budget_attr="d_dead", budget=100.0,
                     cost_attr="time", use_heuristic=True):
    """Constrained shortest path (min cost_attr s.t. sum budget_attr <= budget),
    guided by the additive tile heuristic on cost_attr.

    Returns (path, total_cost, total_budget, expanded, runtime_ms).
    """
    t0 = _time.perf_counter()

    # --- admissible tile heuristic on the COST dimension (time) ---
    if use_heuristic:
        dest_tile = part.tile_of[d]
        edge_cost = lambda u, v, k: float(G[u][v][k][cost_attr])
        Htile = build_tile_heuristic(part, dest_tile, edge_cost, "additive")
        def h(node):
            return Htile.get(part.tile_of[node], 0.0)   # lower bound on remaining cost
    else:
        def h(node):
            return 0.0

    counter = itertools.count()
    best = {s: [(0.0, 0.0)]}                 # node -> [(cost, budget)] non-dominated
    prev = {}
    # priority = f = cost + h(node); tie-break budget then counter
    pq = [(h(s), 0.0, 0.0, next(counter), s)]
    expanded = 0
    goal = None

    while pq:
        f, cost, bud, _, u = heapq.heappop(pq)
        if not any(abs(c - cost) < 1e-9 and abs(b - bud) < 1e-9
                   for (c, b) in best.get(u, [])):
            continue
        expanded += 1
        if u == d:
            goal = (cost, bud)
            break
        for _, v, k, ed in G.edges(u, keys=True, data=True):
            nb = bud + float(ed[budget_attr])
            if nb > budget:                  # prune: over budget
                continue
            nc = cost + float(ed[cost_attr])
            labels = best.setdefault(v, [])
            if any(c <= nc + 1e-9 and b <= nb + 1e-9 for (c, b) in labels):
                continue
            labels[:] = [(c, b) for (c, b) in labels
                         if not (nc <= c + 1e-9 and nb <= b + 1e-9)]
            labels.append((nc, nb))
            prev[(v, round(nc, 6), round(nb, 6))] = (u, round(cost, 6), round(bud, 6))
            heapq.heappush(pq, (nc + h(v), nc, nb, next(counter), v))

    rt = (_time.perf_counter() - t0) * 1000.0
    if goal is None:
        return [], None, None, expanded, rt

    cost, bud = goal
    path = [d]; state = (d, round(cost, 6), round(bud, 6))
    while state[0] != s:
        p = prev.get(state)
        if p is None:
            break
        path.append(p[0]); state = p
    path.reverse()
    return path, cost, bud, expanded, rt