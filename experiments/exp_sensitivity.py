"""Sensitivity sweep (RQ4) — does the trade-off knee survive different coverage regimes?

For each regime (a value of theta and/or cell-edge params), rebuild the coverage
map + objectives, then run the trade-off sweep over the same OD pairs. Overlay all
regimes' trade-off curves on one figure to show the result is robust.
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
# each regime overrides some signal-config fields; label -> overrides
REGIMES = {
    "light (theta=5)":  {"theta_db": 5.0},
    "moderate (theta=10)": {"theta_db": 10.0},     # the calibrated baseline
    "heavy (theta=15)": {"theta_db": 15.0},
}
N_PAIRS = 500                      # fewer than headline (we do it x3 regimes)
MIN_OD_METERS = 2000.0
MIN_FAST_DEAD = 100.0
# per-regime budget sweep: heavy (34% dead) has too few feasible pairs below
# 300m for a stable median, so its tight/mid budgets are excluded rather than
# reported on an unstable, small population.
    # BUDGETS_BY_REGIME = {
    #     "light (theta=5)":    [25, 50, 100, 200, 300, 400, 600, 800],
    #     "moderate (theta=10)": [25, 50, 100, 200, 300, 400, 600, 800],
    #     "heavy (theta=15)":   [300, 400, 600, 800],
    # }

BUDGETS_BY_REGIME = {
    "light (theta=5)":    [100,200, 400, 600, 800],
    "moderate (theta=10)": [100,200, 400, 600, 800],
    "heavy (theta=15)":   [100,200, 400, 600, 800],
}
MIN_N_FOR_MEDIAN = 20   # warn if a budget point rests on fewer than this many pairs
MAX_TRIES = 30000
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
    # ensure node coords are float for straight-line distance
    for _, nd in Gp.nodes(data=True):
        nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])
    return Gp


def sample_hole_pairs(G, base, n_pairs, seed):
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


def tradeoff_for_graph(G, prefs, pairs, budgets):
    base = make_order(prefs, ["time"])
    rows = []
    for (s, d) in pairs:
        fp, _ = lexico_route(G, s, d, base)
        if not fp:
            continue
        fd, ft = path_dead_time(G, fp)
        if fd < 1.0:
            continue
        for B in budgets:
            cp, _, _ = constrained_route(G, s, d, budget_attr="d_dead",
                                         budget=float(B), cost_attr="time")
            if not cp:
                continue
            cd, ct = path_dead_time(G, cp)
            rows.append({
                "budget": B,
                "dead_removed_pct": 100.0 * (fd - cd) / fd if fd else 0.0,
                "detour_pct": 100.0 * (ct - ft) / ft if ft else 0.0,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["budget", "detour_med", "dead_removed_pct_med", "n"])
    return df.groupby("budget").agg(
        detour_med=("detour_pct", "median"),
        dead_removed_pct_med=("dead_removed_pct", "median"),
        n=("detour_pct", "count"),
    ).reset_index()


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

        # dead fraction of this regime (for the legend / reporting)
        dead_frac = np.mean([1.0 if float(d["dead_fraction"]) >= 0.999 else 0.0
                             for _, _, d in Gp.edges(data=True)])

        base = make_order(prefs, ["time"])
        pairs = sample_hole_pairs(Gp, base, N_PAIRS, cfg.seed)
        budgets = BUDGETS_BY_REGIME[label]
        agg = tradeoff_for_graph(Gp, prefs, pairs, budgets)
        agg["regime"] = label
        all_agg.append(agg)

        ax.plot(agg["detour_med"], agg["dead_removed_pct_med"], "-o",
                color=PALETTE[i], markersize=4, linewidth=1.4,
                label=f"{label}, {dead_frac*100:.0f}\\% dead", zorder=3)
        tqdm.write(f"\n{label}  (fully-dead {dead_frac*100:.0f}%, {len(pairs)} pairs)")
        for _, r in agg.iterrows():
            flag = "  <-- LOW n" if r["n"] < MIN_N_FOR_MEDIAN else ""
            tqdm.write(f"    B={r['budget']:>4.0f}m  removed={r['dead_removed_pct_med']:5.1f}%  "
                       f"detour={r['detour_med']:5.1f}%  (n={int(r['n'])}){flag}")

    ax.set_xlabel(r"Travel-time detour (\%)")
    ax.set_ylabel(r"Dead-zone exposure removed (\%)")
    ax.legend(loc="lower right", fontsize=7)
    grid_box()
    save("sensitivity_tradeoff", category="sensitivity", fig=fig)
    plt.close(fig)

    pd.concat(all_agg).to_csv(f"results/tables/sensitivity_{stamp}.csv", index=False)
    print(f"\nsaved sensitivity figure + table ({stamp})")