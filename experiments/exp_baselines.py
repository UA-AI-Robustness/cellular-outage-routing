"""Baseline comparison: our constrained method vs weighted-sum coverage-aware
(the competitor), plus shortest/fastest references. Plots both trade-off curves
on one axis to show domination. Uses the parallel harness."""
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.experiment.parallel import run_parallel
from connroute.viz.style import apply_style, grid_box, BLUE, RED, GREY, save

N_PAIRS = 500
MIN_OD_METERS = 2000.0


def sample_pairs():
    cfg = load_config()
    G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
    for _, nd in G.nodes(data=True):
        nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])
    rng = random.Random(cfg.seed); nodes = list(G.nodes()); pairs = []
    while len(pairs) < N_PAIRS:
        s, d = rng.choice(nodes), rng.choice(nodes)
        if s == d: continue
        if np.hypot(G.nodes[s]["x"]-G.nodes[d]["x"], G.nodes[s]["y"]-G.nodes[d]["y"]) >= MIN_OD_METERS:
            pairs.append((s, d))
    return pairs


if __name__ == "__main__":
    pairs = sample_pairs()
    print(f"running {len(pairs)} pairs (baselines) in parallel...")
    rows = run_parallel("experiments.workers:baselines_pair", pairs, desc="pairs")
    df = pd.DataFrame(rows)

    # aggregate each method's curve: median detour & dead-removed per knob value
    def curve(method):
        sub = df[df["method"] == method]
        agg = sub.groupby("knob").agg(
            detour=("detour_pct", "median"),
            removed=("dead_removed_pct", "median")).reset_index()
        return agg.sort_values("detour")

    con = curve("constrained")
    ws = curve("weighted_sum")

    print("\n=== constrained (ours) ===")
    for _, r in con.iterrows():
        print(f"  B={r['knob']:>5.0f} : removed={r['removed']:5.1f}%  detour={r['detour']:5.1f}%")
    print("=== weighted-sum (baseline) ===")
    for _, r in ws.iterrows():
        print(f"  lam={r['knob']:>5.1f}: removed={r['removed']:5.1f}%  detour={r['detour']:5.1f}%")

    # ---- figure: both trade-off curves ----
    apply_style(usetex=True)
    fig, ax = plt.subplots()
    ax.plot(ws["detour"], ws["removed"], "-s", color=GREY, markersize=4,
            linewidth=1.4, label="Weighted-sum (baseline)", zorder=3)
    ax.plot(con["detour"], con["removed"], "-o", color=BLUE, markersize=4,
            linewidth=1.6, label="Constrained (ours)", zorder=4)
    ax.set_xlabel(r"Travel-time detour (\%)")
    ax.set_ylabel(r"Dead-zone exposure removed (\%)")
    ax.legend(loc="lower right", fontsize=7)
    grid_box()
    save("baselines_tradeoff", category="baselines", fig=fig)
    plt.close(fig)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/tables/baselines_{stamp}.csv", index=False)
    print(f"\nsaved baselines table + figure ({stamp})")