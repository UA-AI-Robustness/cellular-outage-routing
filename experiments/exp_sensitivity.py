"""Sensitivity sweep (RQ4) — does the trade-off knee survive different coverage regimes?

For each regime (a value of theta), rebuild the coverage map + objectives, sample
a pool of OD pairs, then restrict to the COMMON subset feasible at every tested
budget. Using the same fixed population at every budget guarantees a monotonic
curve (no population-composition artifacts), at the cost of a smaller n than the
full sampled pool. The common-set size is reported honestly, not forced to a
target value.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from connroute.config import load_config
from connroute.graph.build import build_or_load
from connroute.signal.towers import load_towers
from connroute.signal.layer import build_signal_layer
from connroute.objectives.attach import attach_objectives
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route, constrained_route
from connroute.viz.style import apply_style, grid_box, PALETTE, save

# ---- sweep parameters ----
REGIMES = {
    "light (theta=5)":  {"theta_db": 5.0},
    "moderate (theta=10)": {"theta_db": 10.0},
    "heavy (theta=15)": {"theta_db": 15.0},
}
BUDGETS = [25, 50, 100, 200, 300, 400, 600, 800, 1200, 1600]
POOL_SIZE = 2000          # larger raw pool, since we need pairs feasible at EVERY budget
MIN_OD_METERS = 2000.0
MIN_FAST_DEAD = 100.0
MIN_COMMON_SET = 25       # warn if the common-feasible set is smaller than this
MAX_TRIES = 60000
TOWER_FILE = "data/raw/opencellid_us_310.csv"


def straight_line_m(G, s, d):
    return float(np.hypot(G.nodes[s]["x"] - G.nodes[d]["x"],
                          G.nodes[s]["y"] - G.nodes[d]["y"]))


def path_dead_time(G, path):
    dd = tt = 0.0
    for a, b in zip(path[:-1], path[1:]):
        ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
        dd += float(ed["d_dead"]); tt += float(ed["time"])
    return dd, tt


def build_objectives_for_regime(cfg, towers, G_raw, overrides):
    """Rebuild signal + objectives with signal-config overrides applied."""
    cfg2 = deepcopy(cfg)
    for key, val in overrides.items():
        setattr(cfg2.signal, key, val)
    Gp = build_signal_layer(cfg2, towers, G_raw, verbose=False)
    Gp = attach_objectives(cfg2, Gp)
    for _, nd in Gp.nodes(data=True):
        nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])
    return Gp


def sample_hole_pairs(G, base, n_pairs, seed):
    """Sample a large pool of 'eligible' pairs (fastest route crosses a hole)."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    pairs, tries = [], 0
    while len(pairs) < n_pairs and tries < MAX_TRIES:
        tries += 1
        s, d = rng.choice(nodes), rng.choice(nodes)
        if s == d or straight_line_m(G, s, d) < MIN_OD_METERS:
            continue
        fp, _ = lexico_route(G, s, d, base)
        if not fp:
            continue
        fd, ft = path_dead_time(G, fp)
        if fd < MIN_FAST_DEAD:
            continue
        pairs.append((s, d))
    return pairs


def common_feasible_tradeoff(G, prefs, pairs, budgets):
    """For each budget, find feasible pairs; then restrict ALL budgets to the
    intersection (pairs feasible at every tested budget). Returns the
    common-set trade-off table and the common-set size."""
    base = make_order(prefs, ["time"])

    # first pass: compute per-pair, per-budget feasibility + metrics
    per_pair_budget = {}   # (pair, B) -> (dead_removed_pct, detour_pct)
    fast_cache = {}
    feasible_sets = {B: set() for B in budgets}

    for (s, d) in pairs:
        fp, _ = lexico_route(G, s, d, base)
        if not fp:
            continue
        fd, ft = path_dead_time(G, fp)
        if fd < 1.0:
            continue
        fast_cache[(s, d)] = (fd, ft)
        for B in budgets:
            cp, _, _ = constrained_route(G, s, d, budget_attr="d_dead",
                                         budget=float(B), cost_attr="time")
            if not cp:
                continue
            cd, ct = path_dead_time(G, cp)
            per_pair_budget[((s, d), B)] = (
                100.0 * (fd - cd) / fd if fd else 0.0,
                100.0 * (ct - ft) / ft if ft else 0.0,
            )
            feasible_sets[B].add((s, d))

    # common set: pairs feasible at EVERY tested budget
    common = set.intersection(*feasible_sets.values()) if feasible_sets else set()
    common_n = len(common)

    rows = []
    for B in budgets:
        vals_removed = [per_pair_budget[(p, B)][0] for p in common]
        vals_detour = [per_pair_budget[(p, B)][1] for p in common]
        if not vals_removed:
            continue
        rows.append({
            "budget": B,
            "detour_med": float(np.median(vals_detour)),
            "dead_removed_pct_med": float(np.median(vals_removed)),
            "n": common_n,
        })
    return pd.DataFrame(rows), common_n


if __name__ == "__main__":
    cfg = load_config()
    prefs = build_preferences(cfg)
    towers = load_towers(TOWER_FILE)
    G_raw = build_or_load(cfg)

    apply_style(usetex=True)
    fig, ax = plt.subplots()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    all_agg = []

    for i, (label, overrides) in enumerate(tqdm(REGIMES.items(), desc="regimes", unit="regime")):
        Gp = build_objectives_for_regime(cfg, towers, G_raw, overrides)

        dead_frac = np.mean([1.0 if float(d["dead_fraction"]) >= 0.999 else 0.0
                             for _, _, d in Gp.edges(data=True)])

        base = make_order(prefs, ["time"])
        pairs = sample_hole_pairs(Gp, base, POOL_SIZE, cfg.seed)
        agg, common_n = common_feasible_tradeoff(Gp, prefs, pairs, BUDGETS)
        agg["regime"] = label
        all_agg.append(agg)

        ax.plot(agg["detour_med"], agg["dead_removed_pct_med"], "-o",
                color=PALETTE[i], markersize=4, linewidth=1.4,
                label=f"{label}, {dead_frac*100:.0f}\\% dead (n={common_n})", zorder=3)

        tqdm.write(f"\n{label}  (fully-dead {dead_frac*100:.0f}%, "
                   f"pool={len(pairs)}, common-feasible n={common_n})")
        if common_n < MIN_COMMON_SET:
            tqdm.write(f"  WARNING: common set below {MIN_COMMON_SET} pairs; "
                       f"consider increasing POOL_SIZE for this regime")
        for _, r in agg.iterrows():
            tqdm.write(f"    B={r['budget']:>5.0f}m  removed={r['dead_removed_pct_med']:5.1f}%  "
                       f"detour={r['detour_med']:5.1f}%")

    ax.set_xlabel(r"Travel-time detour (\%)")
    ax.set_ylabel(r"Dead-zone exposure removed (\%)")
    ax.legend(loc="lower right", fontsize=7)
    grid_box()
    save("sensitivity_tradeoff", category="sensitivity", fig=fig)
    plt.close(fig)

    pd.concat(all_agg).to_csv(f"results/tables/sensitivity_{stamp}.csv", index=False)
    print(f"\nsaved sensitivity figure + table ({stamp})")