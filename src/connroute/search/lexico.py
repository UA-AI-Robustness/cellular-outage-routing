"""Stage 3 — routing search.

Two search modes:
  1. lexico_route         : mixed-aggregation lexicographic search. Each preference
                            combines along the route with its own operator (sum/min),
                            compared lexicographically (maximize convention), with an
                            optional per-preference tolerance eps.
  2. constrained_route    : minimize a summed cost (e.g. time) subject to a summed
                            budget (e.g. dead-exposure) <= B. Resource-constrained
                            shortest path. This is the well-posed formulation for
                            "avoid dead zones, but take the fastest route that does".

Everything in lexico_route is in the MAXIMIZE convention: minimized objectives are
stored negated in preferences.py, so 'better' always means 'lexicographically larger'.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import itertools

from connroute.search.preferences import Preference


# ======================================================================
# Lexicographic mixed-aggregation search
# ======================================================================

def lex_greater(a: tuple, b: tuple, eps: tuple | None = None) -> bool:
    """True if vector a is lexicographically greater than b (optionally within eps).

    Comparison walks components in priority order; a difference within eps[i] is
    treated as a tie so lower-priority components can decide.
    """
    for i, (ai, bi) in enumerate(zip(a, b)):
        tol = 0.0 if eps is None else eps[i]
        if ai > bi + tol:
            return True
        if ai < bi - tol:
            return False
        # within tolerance -> tie, continue to next component
    return False   # all components tied


def _identity_vec(order: list[Preference]) -> tuple:
    return tuple(p.identity for p in order)


def _sentinel_vec(order: list[Preference]) -> tuple:
    return tuple(p.sentinel for p in order)


def _combine_vec(g: tuple, edge: dict, order: list[Preference]) -> tuple:
    """Extend g by one edge: each component combines with its preference's operator."""
    return tuple(p.combine(g[i], p.contrib(edge)) for i, p in enumerate(order))


@dataclass(order=False)
class _Item:
    """Heap item; ordered so that heapq (a min-heap) yields the lexicographically
    LARGEST g-vector first."""
    key: tuple
    count: int
    node: int

    def __lt__(self, other: "_Item") -> bool:
        # invert: 'smaller' for the min-heap == lexicographically greater key
        return lex_greater(self.key, other.key)


def lexico_route(G, s: int, d: int, order: list[Preference], eps: tuple | None = None):
    """Lexicographically optimal route from s to d under `order`.

    Returns (path, g_vector). path is a list of node ids ([] if unreachable).
    g_vector is in the maximize convention; use readable_scores() to interpret.
    """
    id_vec = _identity_vec(order)
    sent = _sentinel_vec(order)
    best: dict[int, tuple] = {s: id_vec}
    prev: dict[int, int] = {}
    counter = itertools.count()

    pq: list[_Item] = [_Item(id_vec, next(counter), s)]

    while pq:
        item = heapq.heappop(pq)
        u, gu = item.node, item.key

        # stale entry: a better label for u was recorded after this was pushed
        if lex_greater(best.get(u, sent), gu):
            continue
        if u == d:
            break

        for _, v, k, edata in G.edges(u, keys=True, data=True):
            gv = _combine_vec(gu, edata, order)
            cur = best.get(v)
            if cur is None or lex_greater(gv, cur, eps):
                best[v] = gv
                prev[v] = u
                heapq.heappush(pq, _Item(gv, next(counter), v))

    if d not in best:
        return [], None

    # reconstruct
    path = [d]
    while path[-1] != s:
        p = prev.get(path[-1])
        if p is None:
            return [], None
        path.append(p)
    path.reverse()
    return path, best[d]


def readable_scores(g: tuple, order: list[Preference]) -> dict:
    """Turn a raw g-vector (maximize convention) into human-readable per-objective values."""
    out = {}
    for i, p in enumerate(order):
        val = g[i]
        out[p.name] = -val if p.direction == "min" else val
    return out


# ======================================================================
# Constrained shortest path: min summed cost s.t. summed budget <= B
# ======================================================================

def constrained_route(G, s: int, d: int,
                      budget_attr: str = "d_dead", budget: float = 100.0,
                      cost_attr: str = "time"):
    """Minimize total `cost_attr` subject to total `budget_attr` <= `budget`.

    Resource-constrained shortest path via label-setting: each node keeps a set of
    non-dominated (cost, budget) labels; any extension exceeding `budget` is pruned.

    Returns (path, total_cost, total_budget_used), or ([], None, None) if infeasible.
    """
    counter = itertools.count()
    # best[node] = list of non-dominated (cost, budget) labels
    best: dict[int, list[tuple[float, float]]] = {s: [(0.0, 0.0)]}
    prev: dict[tuple, tuple] = {}
    pq: list[tuple] = [(0.0, 0.0, next(counter), s)]   # (cost, budget, tie, node)

    goal_state = None
    while pq:
        cost, bud, _, u = heapq.heappop(pq)

        # is this label still present (non-dominated) at u?
        if not any(abs(c - cost) < 1e-9 and abs(b - bud) < 1e-9
                   for (c, b) in best.get(u, [])):
            continue
        if u == d:
            goal_state = (cost, bud)
            break

        for _, v, k, ed in G.edges(u, keys=True, data=True):
            nb = bud + float(ed[budget_attr])
            if nb > budget:                     # PRUNE: over the budget
                continue
            nc = cost + float(ed[cost_attr])
            labels = best.setdefault(v, [])
            # dominated if an existing label is no worse on both dimensions
            if any(c <= nc + 1e-9 and b <= nb + 1e-9 for (c, b) in labels):
                continue
            # drop labels the new one dominates
            labels[:] = [(c, b) for (c, b) in labels
                         if not (nc <= c + 1e-9 and nb <= b + 1e-9)]
            labels.append((nc, nb))
            prev[(v, round(nc, 6), round(nb, 6))] = (u, round(cost, 6), round(bud, 6))
            heapq.heappush(pq, (nc, nb, next(counter), v))

    if goal_state is None:
        return [], None, None

    # reconstruct
    cost, bud = goal_state
    path = [d]
    state = (d, round(cost, 6), round(bud, 6))
    while state[0] != s:
        p = prev.get(state)
        if p is None:
            break
        path.append(p[0])
        state = p
    path.reverse()
    return path, cost, bud


def constrained_route_astar(G, s: int, d: int,
                            budget_attr: str = "d_dead", budget: float = 100.0,
                            cost_attr: str = "time",
                            max_speed_mps: float | None = None):
    """A*-accelerated constrained shortest path: minimize summed cost_attr subject
    to summed budget_attr <= budget. Uses a straight-line admissible time heuristic.

    Returns identical routes to constrained_route (admissible heuristic), just faster.
    Requires node attributes 'x','y' (projected metres). max_speed_mps is the graph's
    fastest edge speed (used for the admissible time bound); auto-computed if None.
    """
    import heapq, itertools, math

    # --- admissible heuristic: straight-line distance / max speed <= true remaining time ---
    if max_speed_mps is None:
        # fastest edge speed in the graph (m/s); guarantees under-estimate of time
        speeds = []
        for _, _, ed in G.edges(data=True):
            L = float(ed.get("length", 0.0)); T = float(ed.get(cost_attr, 0.0))
            if T > 0 and L > 0:
                speeds.append(L / T)
        max_speed_mps = max(speeds) if speeds else 30.0

    dx_goal, dy_goal = float(G.nodes[d]["x"]), float(G.nodes[d]["y"])
    def h(u):
        ux, uy = float(G.nodes[u]["x"]), float(G.nodes[u]["y"])
        return math.hypot(ux - dx_goal, uy - dy_goal) / max_speed_mps   # <= true remaining time

    counter = itertools.count()
    best: dict[int, list[tuple[float, float]]] = {s: [(0.0, 0.0)]}
    prev: dict[tuple, tuple] = {}
    # priority = f = g_cost + h(node); tie-break by budget then counter
    pq = [(h(s), 0.0, 0.0, next(counter), s)]   # (f, cost, budget, tie, node)

    goal_state = None
    while pq:
        f, cost, bud, _, u = heapq.heappop(pq)
        if not any(abs(c - cost) < 1e-9 and abs(b - bud) < 1e-9
                   for (c, b) in best.get(u, [])):
            continue
        if u == d:
            goal_state = (cost, bud)
            break
        for _, v, k, ed in G.edges(u, keys=True, data=True):
            nb = bud + float(ed[budget_attr])
            if nb > budget:
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

    if goal_state is None:
        return [], None, None

    cost, bud = goal_state
    path = [d]; state = (d, round(cost, 6), round(bud, 6))
    while state[0] != s:
        p = prev.get(state)
        if p is None:
            break
        path.append(p[0]); state = p
    path.reverse()
    return path, cost, bud

def weighted_sum_route(G, s, d, lam=0.0, cost_attr="time", penalty_attr="d_dead"):
    """Baseline: minimize sum of (cost_attr + lam * penalty_attr) over the route.
    This is the standard coverage-aware scalarization (the competitor's approach).
    Plain Dijkstra on the blended edge weight."""
    import heapq, itertools
    counter = itertools.count()
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, next(counter), s)]
    while pq:
        dcur, _, u = heapq.heappop(pq)
        if dcur > dist.get(u, float("inf")):
            continue
        if u == d:
            break
        for _, v, k, ed in G.edges(u, keys=True, data=True):
            w = float(ed[cost_attr]) + lam * float(ed[penalty_attr])
            nd = dcur + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, next(counter), v))
    if d not in dist:
        return []
    path = [d]
    while path[-1] != s:
        p = prev.get(path[-1])
        if p is None:
            return []
        path.append(p)
    path.reverse()
    return path

def min_rate_route(G, s, d, tau=0.0, cost_attr="time", rate_attr="q"):
    """Live-call framing: minimize summed cost_attr over routes where EVERY edge
    has rate_attr >= tau (guaranteed worst-segment rate). Plain Dijkstra on time,
    with edges below tau forbidden. Returns [] if no such route exists."""
    import heapq, itertools
    counter = itertools.count()
    dist = {s: 0.0}; prev = {}
    pq = [(0.0, next(counter), s)]
    while pq:
        dcur, _, u = heapq.heappop(pq)
        if dcur > dist.get(u, float("inf")):
            continue
        if u == d:
            break
        for _, v, k, ed in G.edges(u, keys=True, data=True):
            if float(ed[rate_attr]) < tau:      # FORBID edges below the rate floor
                continue
            nd = dcur + float(ed[cost_attr])
            if nd < dist.get(v, float("inf")):
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, next(counter), v))
    if d not in dist:
        return []
    path = [d]
    while path[-1] != s:
        p = prev.get(path[-1])
        if p is None: return []
        path.append(p)
    path.reverse()
    return path

def maxmin_rate_within_time(G, s, d, time_budget, cost_attr="time", rate_attr="q"):
    """Maximize the worst-segment (bottleneck) rate over routes whose total
    cost_attr <= time_budget. Returns (path, worst_rate, total_time) or ([],None,None).

    Method: binary-search the achievable rate floor. For a candidate floor tau,
    forbid edges below tau and check if a route within time_budget exists (Dijkstra
    on time). The largest feasible tau is the answer.
    """
    import heapq, itertools

    def fastest_time_with_floor(tau):
        """Min total time over routes using only edges with rate >= tau; inf if none."""
        counter = itertools.count()
        dist = {s: 0.0}
        pq = [(0.0, next(counter), s)]
        while pq:
            dcur, _, u = heapq.heappop(pq)
            if dcur > dist.get(u, float("inf")):
                continue
            if u == d:
                return dcur
            for _, v, k, ed in G.edges(u, keys=True, data=True):
                if float(ed[rate_attr]) < tau:
                    continue
                nd = dcur + float(ed[cost_attr])
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, next(counter), v))
        return dist.get(d, float("inf"))

    # candidate rate floors = the distinct edge-rate values on the graph, sorted.
    # binary-search the highest floor whose fastest route fits the time budget.
    rates = sorted({round(float(ed[rate_attr]), 4) for _, _, ed in G.edges(data=True)})
    lo, hi, best_tau = 0, len(rates) - 1, 0.0
    # feasibility at tau=0 is required (else no route at all)
    if fastest_time_with_floor(0.0) > time_budget:
        return [], None, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if fastest_time_with_floor(rates[mid]) <= time_budget:
            best_tau = rates[mid]
            lo = mid + 1
        else:
            hi = mid - 1

    # reconstruct the actual route at best_tau (fastest within floor)
    counter = itertools.count()
    dist = {s: 0.0}; prev = {}
    pq = [(0.0, next(counter), s)]
    while pq:
        dcur, _, u = heapq.heappop(pq)
        if dcur > dist.get(u, float("inf")):
            continue
        if u == d:
            break
        for _, v, k, ed in G.edges(u, keys=True, data=True):
            if float(ed[rate_attr]) < best_tau:
                continue
            nd = dcur + float(ed[cost_attr])
            if nd < dist.get(v, float("inf")):
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, next(counter), v))
    if d not in dist:
        return [], None, None
    path = [d]
    while path[-1] != s:
        p = prev.get(path[-1])
        if p is None: return [], None, None
        path.append(p)
    path.reverse()
    return path, best_tau, dist[d]

def max_data_within_time(G, s, d, time_budget,
                         cost_attr="time", data_attr="q_dwell"):
    """Maximize summed data_attr subject to summed cost_attr <= time_budget.
    Resource-constrained: label = (data_so_far, time_so_far); keep non-dominated
    labels (more data AND less time dominates); prune labels over the time budget.
    Returns (path, total_data, total_time) or ([], None, None)."""
    import heapq, itertools
    counter = itertools.count()
    # maximize data -> store NEGATIVE data in the heap so heapq (min-heap) pops best first
    best: dict[int, list[tuple[float, float]]] = {s: [(0.0, 0.0)]}  # (data, time)
    prev: dict[tuple, tuple] = {}
    pq = [(-0.0, 0.0, next(counter), s)]   # (-data, time, tie, node)

    goal = None
    while pq:
        neg_data, t, _, u = heapq.heappop(pq)
        data = -neg_data
        if not any(abs(dd - data) < 1e-9 and abs(tt - t) < 1e-9
                   for (dd, tt) in best.get(u, [])):
            continue
        if u == d:
            goal = (data, t)
            break
        for _, v, k, ed in G.edges(u, keys=True, data=True):
            nt = t + float(ed[cost_attr])
            if nt > time_budget:            # PRUNE: over the time budget
                continue
            nd = data + float(ed[data_attr])
            labels = best.setdefault(v, [])
            # dominated if an existing label has >= data AND <= time
            if any(dd >= nd - 1e-9 and tt <= nt + 1e-9 for (dd, tt) in labels):
                continue
            labels[:] = [(dd, tt) for (dd, tt) in labels
                         if not (nd >= dd - 1e-9 and nt <= tt + 1e-9)]
            labels.append((nd, nt))
            prev[(v, round(nd, 6), round(nt, 6))] = (u, round(data, 6), round(t, 6))
            heapq.heappush(pq, (-nd, nt, next(counter), v))

    if goal is None:
        return [], None, None
    data, t = goal
    path = [d]; state = (d, round(data, 6), round(t, 6))
    while state[0] != s:
        p = prev.get(state)
        if p is None: break
        path.append(p[0]); state = p
    path.reverse()
    return path, data, t