"""Live-call trade-off (max-min framing).

For each OD pair, and a sweep of allowed detour budgets, find the route that
MAXIMIZES the worst-segment rate (bottleneck) while staying within the time
budget. This gives one consistent population (all pairs) at every point, so the
trade-off is monotonic -- no feasibility collapse, no fold-back.

Story: granting an X% detour raises the guaranteed worst-segment rate from the
fastest route's (low) value up to the best achievable within that detour.
"""
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
from connroute.viz.style import apply_style, grid_box, RED, save

N_PAIRS = 500
MIN_OD_METERS = 2000.0


def sample_pairs():
    cfg = load_config()
    G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
    for _, nd in G.nodes(data=True):
        nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])
    rng = random.Random(cfg.seed)
    nodes = list(G.nodes())
    pairs = []
    while len(pairs) < N_PAIRS:
        s, d = rng.choice(nodes), rng.choice(nodes)
        if s == d:
            continue
        if np.hypot(G.nodes[s]["x"] - G.nodes[d]["x"],
                    G.nodes[s]["y"] - G.nodes[d]["y"]) >= MIN_OD_METERS:
            pairs.append((s, d))
    return pairs


if __name__ == "__main__":
    pairs = sample_pairs()
    print(f"running {len(pairs)} pairs (live-call max-min) in parallel...")
    rows = run_parallel("experiments.workers:livecall_pair", pairs, desc="pairs")
    df = pd.DataFrame(rows)

    # aggregate: same population (all feasible pairs) at every detour budget
    agg = df.groupby("detour_allowed").agg(
        best_qmin=("best_qmin", "median"),
        gain=("qmin_gain", "median"),
        actual_detour=("actual_detour_pct", "median"),
        n=("s", "count"),
    ).reset_index()

    print("\n=== live-call max-min trade-off (median over ALL pairs) ===")
    print(f"{'detour allowed %':>16} {'best worst-rate':>16} {'gain':>8} "
          f"{'actual detour %':>16} {'n':>6}")
    for _, r in agg.iterrows():
        print(f"{r['detour_allowed']:>16.0f} {r['best_qmin']:>16.3f} "
              f"{r['gain']:>8.3f} {r['actual_detour']:>16.1f} {int(r['n']):>6}")

    # ---- figure: best achievable worst-segment rate vs detour ----
    apply_style(usetex=True)
    fig, ax = plt.subplots()
    ax.plot(agg["actual_detour"], agg["best_qmin"], "-o", color=RED,
            markersize=4, linewidth=1.5, zorder=3)
    for _, r in agg.iterrows():
        ax.annotate(f"+{int(r['detour_allowed'])}\\%",
                    (r["actual_detour"], r["best_qmin"]),
                    textcoords="offset points", xytext=(5, -6), fontsize=6)
    ax.set_xlabel(r"Travel-time detour (\%)")
    ax.set_ylabel(r"Best achievable worst-segment rate (norm.)")
    grid_box()
    save("livecall_maxmin", category="livecall", fig=fig)
    plt.close(fig)

    # ---- save table ----
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/tables/livecall_maxmin_{stamp}.csv", index=False)
    agg.to_csv(f"results/tables/livecall_maxmin_agg_{stamp}.csv", index=False)
    print(f"\nsaved live-call max-min table + figure ({stamp})")