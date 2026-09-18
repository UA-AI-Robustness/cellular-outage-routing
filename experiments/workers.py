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