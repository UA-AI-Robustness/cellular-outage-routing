"""Stage 3 — mixed-aggregation lexicographic shortest path.

Finds the route from s to d that is optimal under a priority-ordered list of
Preferences, where each preference combines along the route with its OWN operator
(sum or min), compared lexicographically. Everything is in the MAXIMIZE
convention (minimized objectives were negated in preferences.py), so 'better'
always means 'lexicographically larger g-vector'.

First-pass: Dijkstra-style (H=0 admissible heuristic). Correctness before speed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import heapq
import itertools

from connroute.search.preferences import Preference


# ---- lexicographic comparison on g-vectors (all maximize) ----
def lex_greater(a: tuple, b: tuple, eps: tuple | None = None) -> bool:
    """True if vector a is lexicographically greater than b (optionally within eps)."""
    for i, (ai, bi) in enumerate(zip(a, b)):
        tol = 0.0 if eps is None else eps[i]
        if ai > bi + tol:
            return True
        if ai < bi - tol:
            return False
        # within tol -> treat as tie, move to next component
    return False   # all equal


def _identity_vec(order: list[Preference]) -> tuple:
    return tuple(p.identity for p in order)


def _combine_vec(g: tuple, edge: dict, order: list[Preference]) -> tuple:
    """Extend g by one edge: each component combines with its preference's operator."""
    return tuple(p.combine(g[i], p.contrib(edge)) for i, p in enumerate(order))


@dataclass(order=False)
class _Item:
    """Heap item. We negate for a max-heap via a custom sort key on the g-vector."""
    key: tuple            # the g-vector, compared lexicographically (larger = better)
    count: int            # tie-breaker so heapq never compares nodes
    node: int
    def __lt__(self, other: "_Item") -> bool:
        # heapq is a MIN-heap; we want MAX lexicographic -> invert the comparison
        return lex_greater(self.key, other.key)


def lexico_route(
    G,
    s: int,
    d: int,
    order: list[Preference],
    eps: tuple | None = None,
):
    """Return (path, g_vector) for the lexicographically optimal route s->d.

    path : list of node ids  (empty if d unreachable)
    g    : the achieved g-vector at d (maximize convention; negate minimized comps to read them)
    """
    id_vec = _identity_vec(order)
    best: dict[int, tuple] = {s: id_vec}     # best g-vector found per node
    prev: dict[int, int] = {}
    counter = itertools.count()

    pq: list[_Item] = [_Item(id_vec, next(counter), s)]

    while pq:
        item = heapq.heappop(pq)
        u, gu = item.node, item.key

        # stale entry? (a better label for u was found after this was pushed)
        if lex_greater(best.get(u, tuple(p.sentinel for p in order)), gu):
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
        # minimized objectives were stored negated -> flip back for reporting
        out[p.name] = -val if p.direction == "min" else val
    return out