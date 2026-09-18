"""Per-pair worker functions for parallel experiments. Must be module-level
(picklable). Each takes (G, cfg, task) and returns a list of result-row dicts.
"""
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route, constrained_route


def _dead_time(G, path):
    dd = tt = 0.0
    for a, b in zip(path[:-1], path[1:]):
        ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
        dd += float(ed["d_dead"]); tt += float(ed["time"])
    return dd, tt


def divergence_pair(G, cfg, task):
    """task = (s, d). Returns rows comparing fastest vs constrained continuity."""
    s, d = task
    prefs = build_preferences(cfg)
    base = make_order(prefs, ["time"])
    fp, _ = lexico_route(G, s, d, base)
    if not fp:
        return []
    fd, ft = _dead_time(G, fp)
    rows = []
    for B in (50, 100, 200, 400):
        cp, _, _ = constrained_route(G, s, d, budget_attr="d_dead",
                                     budget=float(B), cost_attr="time")
        if not cp:
            continue
        cd, ct = _dead_time(G, cp)
        rows.append({
            "s": s, "d": d, "budget": B,
            "fast_dead": fd, "fast_time": ft,
            "conn_dead": cd, "conn_time": ct,
            "dead_removed_pct": 100.0 * (fd - cd) / fd if fd else 0.0,
            "detour_pct": 100.0 * (ct - ft) / ft if ft else 0.0,
            "diverged": cp != fp,
        })
    return rows


def baselines_pair(G, cfg, task):
    """task = (s, d). Compute fastest, shortest, weighted-sum (sweep lambda),
    and constrained (sweep B). Returns rows for the comparison figure."""
    from connroute.search.lexico import (lexico_route, constrained_route,
                                          weighted_sum_route)
    s, d = task
    prefs = build_preferences(cfg)

    # reference: fastest
    fp, _ = lexico_route(G, s, d, make_order(prefs, ["time"]))
    if not fp:
        return []
    fd, ft = _dead_time(G, fp)
    if fd <= 100:          # only pairs whose fast route crosses a real hole
        return []

    rows = []

    # --- our method: constrained, sweep budget B ---
    for B in (25, 50, 100, 200, 400, 800):
        cp, _, _ = constrained_route(G, s, d, budget=float(B))
        if not cp:
            continue
        cd, ct = _dead_time(G, cp)
        rows.append({"method": "constrained", "knob": B, "s": s, "d": d,
                     "dead_removed_pct": 100.0*(fd-cd)/fd, "detour_pct": 100.0*(ct-ft)/ft})

    # --- baseline: weighted-sum, sweep lambda ---
    for lam in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        wp = weighted_sum_route(G, s, d, lam=lam)
        if not wp:
            continue
        wd, wt = _dead_time(G, wp)
        rows.append({"method": "weighted_sum", "knob": lam, "s": s, "d": d,
                     "dead_removed_pct": 100.0*(fd-wd)/fd, "detour_pct": 100.0*(wt-ft)/ft})

    # --- reference points: shortest ---
    sp, _ = lexico_route(G, s, d, make_order(prefs, ["distance"]))
    if sp:
        sd, st = _dead_time(G, sp)
        rows.append({"method": "shortest", "knob": 0, "s": s, "d": d,
                     "dead_removed_pct": 100.0*(fd-sd)/fd, "detour_pct": 100.0*(st-ft)/ft})

    return rows