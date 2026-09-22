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
    """task = (s, d). Live-call via call-inadequate distance (d_lowrate):
    our constrained method (sweep budget B) vs weighted-sum baseline (sweep lambda).
    Mirrors the continuity/baseline comparison."""
    from connroute.search.lexico import (lexico_route, constrained_route,
                                          weighted_sum_route)
    s, d = task
    prefs = build_preferences(cfg)

    # fastest route = reference
    fp, _ = lexico_route(G, s, d, make_order(prefs, ["time"]))
    if not fp:
        return []

    def path_lowrate_time(path):
        lr = tt = 0.0
        for a, b in zip(path[:-1], path[1:]):
            ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
            lr += float(ed["d_lowrate"]); tt += float(ed["time"])
        return lr, tt

    fast_lr, ft = path_lowrate_time(fp)
    if fast_lr <= 100:            # only pairs whose fast route has real low-rate distance
        return []

    rows = []
    # our method: constrained on low-rate distance, sweep budget
    for B in (25, 50, 100, 200, 400, 800):
        cp, _, _ = constrained_route(G, s, d, budget_attr="d_lowrate",
                                     budget=float(B), cost_attr="time")
        if not cp:
            continue
        clr, ct = path_lowrate_time(cp)
        rows.append({"method": "constrained", "knob": B, "s": s, "d": d,
                     "lowrate_removed_pct": 100.0*(fast_lr-clr)/fast_lr,
                     "detour_pct": 100.0*(ct-ft)/ft if ft else 0.0})

    # baseline: weighted-sum time + lambda * low-rate distance, sweep lambda
    for lam in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        wp = weighted_sum_route(G, s, d, lam=lam, penalty_attr="d_lowrate")
        if not wp:
            continue
        wlr, wt = path_lowrate_time(wp)
        rows.append({"method": "weighted_sum", "knob": lam, "s": s, "d": d,
                     "lowrate_removed_pct": 100.0*(fast_lr-wlr)/fast_lr,
                     "detour_pct": 100.0*(wt-ft)/ft if ft else 0.0})
    return rows




def upload_pair(G, cfg, task):
    """task = (s, d). Upload-adequate routing via d_lowupload (Option 1):
    constrained method (sweep budget) vs weighted-sum baseline (sweep lambda)."""
    from connroute.search.lexico import (lexico_route, constrained_route,
                                          weighted_sum_route)
    s, d = task
    prefs = build_preferences(cfg)

    fp, _ = lexico_route(G, s, d, make_order(prefs, ["time"]))
    if not fp:
        return []

    def path_lowupload_time(path):
        lu = tt = 0.0
        for a, b in zip(path[:-1], path[1:]):
            ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
            lu += float(ed["d_lowupload"]); tt += float(ed["time"])
        return lu, tt

    fast_lu, ft = path_lowupload_time(fp)
    if fast_lu <= 100:            # only pairs whose fast route has real upload-inadequate distance
        return []

    rows = []
    for B in (50, 100, 200, 400, 800, 1600):
        cp, _, _ = constrained_route(G, s, d, budget_attr="d_lowupload",
                                     budget=float(B), cost_attr="time")
        if not cp:
            continue
        clu, ct = path_lowupload_time(cp)
        rows.append({"method": "constrained", "knob": B, "s": s, "d": d,
                     "lowupload_removed_pct": 100.0*(fast_lu-clu)/fast_lu,
                     "detour_pct": 100.0*(ct-ft)/ft if ft else 0.0})

    for lam in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        wp = weighted_sum_route(G, s, d, lam=lam, penalty_attr="d_lowupload")
        if not wp:
            continue
        wlu, wt = path_lowupload_time(wp)
        rows.append({"method": "weighted_sum", "knob": lam, "s": s, "d": d,
                     "lowupload_removed_pct": 100.0*(fast_lu-wlu)/fast_lu,
                     "detour_pct": 100.0*(wt-ft)/ft if ft else 0.0})
    return rows

def _setup_baruffa(G, cfg):
    """Per-process: attach Baruffa's radio weight R to edges once."""
    from connroute.search.baruffa import compute_radio_weight
    compute_radio_weight(G, kind="on_off")

_R_DONE = False
def baruffa_pair(G, cfg, task):
    """task = (s, d). Ours (constrained, sweep B) vs Baruffa (sweep alpha),
    plus a shortest reference, on a common eligible pair. Exposure reduction is
    vs. the fastest route; detour is travel-time detour.
    Attaches Baruffa's radio weight R once per worker process."""
    global _R_DONE
    if not _R_DONE:
        from connroute.search.baruffa import compute_radio_weight
        compute_radio_weight(G, kind="on_off")
        _R_DONE = True

    from connroute.search.lexico import lexico_route, constrained_route
    from connroute.search.baruffa import baruffa_route
    from connroute.search.preferences import build_preferences, make_order
    s, d = task
    prefs = build_preferences(cfg)

    def dead_time(path):
        dd = tt = 0.0
        for a, b in zip(path[:-1], path[1:]):
            ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
            dd += float(ed["d_dead"]); tt += float(ed["time"])
        return dd, tt

    fp, _ = lexico_route(G, s, d, make_order(prefs, ["time"]))
    if not fp:
        return []
    fast_dead, fast_time = dead_time(fp)
    if fast_dead <= 0:          # eligible pairs only (fastest route crosses a hole)
        return []

    rows = []

    def reduction(dead):
        return 100.0 * (fast_dead - dead) / fast_dead

    def detour(t):
        return 100.0 * (t - fast_time) / fast_time if fast_time else 0.0

    # shortest reference
    sp, _ = lexico_route(G, s, d, make_order(prefs, ["distance"]))
    if sp:
        sd, st = dead_time(sp)
        rows.append({"method": "shortest", "knob": 0, "s": s, "d": d,
                     "reduction": reduction(sd), "detour": detour(st)})

    # ours: constrained, sweep budget
    for B in (25, 50, 100, 200, 400, 800):
        cp, _, _ = constrained_route(G, s, d, budget_attr="d_dead",
                                     budget=float(B), cost_attr="time")
        if not cp:
            continue
        cd, ct = dead_time(cp)
        rows.append({"method": "constrained", "knob": B, "s": s, "d": d,
                     "reduction": reduction(cd), "detour": detour(ct)})

    # Baruffa: radio-discount cost, sweep alpha (R attached above)
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0):
        bp, _ = baruffa_route(G, s, d, alpha=alpha)
        if not bp:
            continue
        bd, bt = dead_time(bp)
        rows.append({"method": "baruffa", "knob": alpha, "s": s, "d": d,
                     "reduction": reduction(bd), "detour": detour(bt)})

    # k-shortest reranking baseline (rerank the k fastest routes by exposure)
    from connroute.search.kshortest import kshortest_rerank
    kp, kexp, kt = kshortest_rerank(G, s, d, k=5)
    if kp:
        rows.append({"method": "kshortest_k5", "knob": 5, "s": s, "d": d,
                     "reduction": reduction(kexp), "detour": detour(kt)})

    return rows

def loadcompare_pair(G, cfg, task):
    """task = (s, d). Rate-inadequate routing under modeled load (d_lo_model)
    vs traffic-informed load (d_lo_traf), sweep budget. Same pairs, same map;
    only the load model differs."""
    from connroute.search.lexico import lexico_route, constrained_route
    from connroute.search.preferences import build_preferences, make_order
    s, d = task
    prefs = build_preferences(cfg)

    fp, _ = lexico_route(G, s, d, make_order(prefs, ["time"]))
    if not fp:
        return []

    def path_metrics(path, attr):
        bad = tt = 0.0
        for a, b in zip(path[:-1], path[1:]):
            ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
            bad += float(ed[attr]); tt += float(ed["time"])
        return bad, tt

    _, ft = path_metrics(fp, "d_lo_model")
    rows = []
    for B in (100, 200, 400, 800, 1600):
        for tag, attr in (("model", "d_lo_model"), ("traffic", "d_lo_traf")):
            cp, _, _ = constrained_route(G, s, d, budget_attr=attr,
                                         budget=float(B), cost_attr="time")
            if not cp:
                continue
            bad, ct = path_metrics(cp, attr)
            # reduction vs fastest, measured under the SAME load model
            fast_bad, _ = path_metrics(fp, attr)
            rows.append({"load": tag, "knob": B, "s": s, "d": d,
                         "removed_pct": 100.0*(fast_bad-bad)/fast_bad if fast_bad > 0 else 0.0,
                         "detour_pct": 100.0*(ct-ft)/ft if ft else 0.0})
    # divergence flag: do the two load models pick different routes at B=200?
    m,_,_ = constrained_route(G, s, d, budget_attr="d_lo_model", budget=200.0)
    t,_,_ = constrained_route(G, s, d, budget_attr="d_lo_traf",  budget=200.0)
    if m and t:
        rows.append({"load": "diverge", "knob": 200, "s": s, "d": d,
                     "removed_pct": 100.0 if m != t else 0.0, "detour_pct": 0.0})
    return rows