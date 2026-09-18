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

def livecall_pair(G, cfg, task):
    """task = (s, d). Max-min framing: for a sweep of detour budgets, the best
    worst-segment rate achievable within that detour. One consistent population."""
    from connroute.search.lexico import lexico_route, maxmin_rate_within_time
    s, d = task
    prefs = build_preferences(cfg)
    fp, _ = lexico_route(G, s, d, make_order(prefs, ["time"]))
    if not fp:
        return []

    def worst_rate(path):
        qmin = float("inf")
        for a, b in zip(path[:-1], path[1:]):
            ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
            qmin = min(qmin, float(ed["q"]))
        return 0.0 if qmin == float("inf") else qmin

    # fastest route's time and its (poor) worst-segment rate
    ft = 0.0
    for a, b in zip(fp[:-1], fp[1:]):
        ft += min(float(e["time"]) for e in G[a][b].values())
    fast_qmin = worst_rate(fp)

    rows = []
    for detour_frac in (0.0, 0.05, 0.10, 0.20, 0.40):
        budget = ft * (1.0 + detour_frac)
        path, tau, tt = maxmin_rate_within_time(G, s, d, time_budget=budget)
        if not path:
            continue
        rows.append({
            "s": s, "d": d, "detour_allowed": detour_frac * 100,
            "fast_qmin": fast_qmin,
            "best_qmin": tau,
            "qmin_gain": tau - fast_qmin,
            "actual_detour_pct": 100.0 * (tt - ft) / ft if ft else 0.0,
        })
    return rows

def upload_pair(G, cfg, task):
    """task = (s, d). Max delivered data within a sweep of detour budgets."""
    from connroute.search.lexico import lexico_route, max_data_within_time
    s, d = task
    prefs = build_preferences(cfg)
    fp, _ = lexico_route(G, s, d, make_order(prefs, ["time"]))
    if not fp:
        return []

    def path_data_time(path):
        dat = tt = 0.0
        for a, b in zip(path[:-1], path[1:]):
            ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
            dat += float(ed["q_dwell"]); tt += float(ed["time"])
        return dat, tt

    fast_data, ft = path_data_time(fp)
    rows = []
    for detour_frac in (0.0, 0.05, 0.10, 0.20, 0.40):
        budget = ft * (1.0 + detour_frac)
        path, data, tt = max_data_within_time(G, s, d, time_budget=budget)
        if not path:
            continue
        rows.append({
            "s": s, "d": d, "detour_allowed": detour_frac * 100,
            "fast_data": fast_data, "best_data": data,
            "data_gain_pct": 100.0 * (data - fast_data) / fast_data if fast_data > 0 else 0.0,
            "actual_detour_pct": 100.0 * (tt - ft) / ft if ft else 0.0,
        })
    return rows