"""Upload trade-off: max delivered data within a detour budget. Parallel."""
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
from connroute.viz.style import apply_style, grid_box, GREEN, save

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
    print(f"running {len(pairs)} pairs (upload) in parallel...")
    rows = run_parallel("experiments.workers:upload_pair", pairs, desc="pairs")
    df = pd.DataFrame(rows)

    agg = df.groupby("detour_allowed").agg(
        data_gain=("data_gain_pct", "median"),
        actual_detour=("actual_detour_pct", "median"),
        n=("s", "count")).reset_index()

    print("\n=== upload trade-off (median over ALL pairs) ===")
    print(f"{'detour allowed %':>16} {'data gain %':>12} {'actual detour %':>16} {'n':>6}")
    for _, r in agg.iterrows():
        print(f"{r['detour_allowed']:>16.0f} {r['data_gain']:>12.1f} {r['actual_detour']:>16.1f} {int(r['n']):>6}")

    apply_style(usetex=True)
    fig, ax = plt.subplots()
    ax.plot(agg["actual_detour"], agg["data_gain"], "-o", color=GREEN, markersize=4, linewidth=1.5, zorder=3)
    for _, r in agg.iterrows():
        ax.annotate(f"+{int(r['detour_allowed'])}\\%", (r["actual_detour"], r["data_gain"]),
                    textcoords="offset points", xytext=(5,-6), fontsize=6)
    ax.set_xlabel(r"Travel-time detour (\%)")
    ax.set_ylabel(r"Delivered data gain (\%)")
    grid_box()
    save("upload_tradeoff", category="upload", fig=fig)
    plt.close(fig)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/tables/upload_{stamp}.csv", index=False)
    print(f"\nsaved upload table + figure ({stamp})")