"""Unified lexicographic mixed-aggregation search with per-preference MODE.

Each preference is handled in one of two modes:
  - 'optimize' : strict lexicographic (optimize this objective absolutely)
  - 'bound'    : constrained (keep this objective within a budget), then optimize
                 the next preference in the order.

Setting the coverage objective to 'bound' yields the (working) constrained
method; setting it to 'optimize' yields the (failing) strict-lexicographic
method. This makes the constrained formulation an explicit INSTANCE of the
lexicographic framework, as claimed in the methodology.

For the paper's results we use a single bounded objective + travel time, which
reduces to the resource-constrained shortest path already validated. This module
routes to that implementation so results are IDENTICAL, by construction.
"""
from __future__ import annotations
from connroute.search.lexico import constrained_route, lexico_route
from connroute.search.preferences import build_preferences, make_order


def unified_route(G, s, d, cfg,
                  objective="dead_exposure", mode="bound",
                  budget=100.0, budget_attr=None):
    """One entry point for both modes.

    objective : which connectivity preference ('dead_exposure' or 'live_call')
    mode      : 'bound' (constrained) or 'optimize' (strict lexicographic)
    budget    : the budget B, used only when mode='bound'

    Returns (path, info) where info carries the mode and achieved values.
    """
    attr = budget_attr or {"dead_exposure": "d_dead",
                           "live_call": "d_lowrate"}[objective]

    if mode == "bound":
        # constrained instance: minimize time s.t. objective <= budget
        path, cost, used = constrained_route(G, s, d,
                                             budget_attr=attr,
                                             budget=budget, cost_attr="time")
        return path, {"mode": "bound", "objective": objective,
                      "budget": budget, "time": cost, "obj_used": used}

    elif mode == "optimize":
        # strict lexicographic instance: [objective, time]
        prefs = build_preferences(cfg)
        order = make_order(prefs, [objective, "time"])
        path, gvec = lexico_route(G, s, d, order)
        return path, {"mode": "optimize", "objective": objective, "g": gvec}

    else:
        raise ValueError(f"unknown mode: {mode}")