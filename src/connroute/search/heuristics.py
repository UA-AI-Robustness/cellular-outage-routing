"""Connectivity-aware heuristics for the constrained search.

The constrained search accumulates TWO quantities: travel time (minimized) and
dead-exposure (constrained <= B). A good heuristic must bound BOTH:

  h_time(u) = a lower bound on remaining travel time  u -> d   (A* direction guidance)
  h_dead(u) = a lower bound on remaining dead-exposure u -> d   (FEASIBILITY pruning)

We compute each EXACTLY by a backward Dijkstra from d on that edge attribute:
the true minimum remaining value is itself a valid (tightest) admissible lower
bound. This is tighter than a coarse tile bound; tiles are the cheap approximation
one would use only if this per-query backward search were too expensive (it isn't
on a city graph). The h_dead bound is the connectivity-aware pruning: if the dead
used so far plus the best-case remaining dead already exceeds B, no continuation
can be feasible, so the branch is cut immediately.
"""
from __future__ import annotations
import heapq
import itertools
import networkx as nx


def backward_lower_bounds(G, d, weight: str) -> dict:
    """Exact minimum summed `weight` from every node to d (following directed edges).

    Computed by Dijkstra from d on the reversed graph. Unreachable nodes are absent
    (treat as +inf). Requires non-negative numeric edge attribute `weight`.
    """
    GR = G.reverse(copy=False)
    return nx.single_source_dijkstra_path_length(GR, d, weight=weight)


def constrained_route_h(G, s: int, d: int,
                        budget_attr: str = "d_dead", budget: float = 100.0,
                        cost_attr: str = "time"):
    """Connectivity-aware A* constrained shortest path.

    Minimize summed cost_attr subject to summed budget_attr <= budget, using:
      - h_dead for feasibility pruning (the connectivity-aware part), and
      - h_time for A* ordering on the cost dimension.
    Returns the SAME optimal cost as constrained_route (both admissible), faster.
    """
    INF = float("inf")
    h_dead = backward_lower_bounds(G, d, budget_attr)
    h_time = backward_lower_bounds(G, d, cost_attr)
    hd = lambda n: h_dead.get(n, INF)
    ht = lambda n: h_time.get(n, INF)

    # if even the best-case remaining dead from s exceeds the budget, infeasible
    if hd(s) > budget:
        return [], None, None

    counter = itertools.count()
    best: dict[int, list[tuple[float, float]]] = {s: [(0.0, 0.0)]}
    prev: dict[tuple, tuple] = {}
    pq = [(ht(s), 0.0, 0.0, next(counter), s)]   # (f=cost+h_time, cost, budget, tie, node)

    goal = None
    while pq:
        f, cost, bud, _, u = heapq.heappop(pq)
        if not any(abs(c - cost) < 1e-9 and abs(b - bud) < 1e-9
                   for (c, b) in best.get(u, [])):
            continue
        if u == d:
            goal = (cost, bud)
            break
        for _, v, k, ed in G.edges(u, keys=True, data=True):
            nb = bud + float(ed[budget_attr])
            # ---- connectivity-aware feasibility pruning ----
            # best-case remaining dead from v is hd(v); if that can't fit, cut now.
            if nb + hd(v) > budget:
                continue
            nc = cost + float(ed[cost_attr])
            labels = best.setdefault(v, [])
            if any(c <= nc + 1e-9 and b <= nb + 1e-9 for (c, b) in labels):
                continue
            labels[:] = [(c, b) for (c, b) in labels
                         if not (nc <= c + 1e-9 and nb <= b + 1e-9)]
            labels.append((nc, nb))
            prev[(v, round(nc, 6), round(nb, 6))] = (u, round(cost, 6), round(bud, 6))
            heapq.heappush(pq, (nc + ht(v), nc, nb, next(counter), v))

    if goal is None:
        return [], None, None

    cost, bud = goal
    path = [d]; state = (d, round(cost, 6), round(bud, 6))
    while state[0] != s:
        p = prev.get(state)
        if p is None:
            break
        path.append(p[0]); state = p
    path.reverse()
    return path, cost, bud