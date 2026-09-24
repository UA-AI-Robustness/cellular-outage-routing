"""Upload-adequate routing trade-off + baseline (Option 1: d_lowupload).
Constrained (sweep budget) vs weighted-sum (sweep lambda), parallel.
Per-budget feasibility (each budget uses its own feasible pairs), with n reported
per row and small samples flagged."""
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
from connroute.viz.style import apply_style, grid_box, BLUE, GREY, save

N_PAIRS = 500
MIN_OD_METERS = 2000.0
BUDGETS = [50, 100, 200, 400, 800, 1600]


def sample_pairs():
    cfg = load_config()
    G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
    for _, nd in G.nodes(data=True):
        nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])
    rng = random.Random(cfg.seed); nodes = list(G.nodes()); pairs = []
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
    print(f"running {len(pairs)} pairs (upload-adequate) in parallel...")
    rows = run_parallel("experiments.workers:upload_pair", pairs, desc="pairs", n_workers=8)
    df = pd.DataFrame(rows)

    # per-budget feasibility: each budget uses ITS OWN feasible pairs
    con_all = df[df["method"] == "constrained"]
    ws_df   = df[df["method"] == "weighted_sum"]
    n_total = df.groupby(["s", "d"]).ngroups   # pairs that produced any row

    def curve(sub):
        return sub.groupby("knob").agg(
            detour=("detour_pct", "median"),
            removed=("lowupload_removed_pct", "median"),
            n=("s", "count")).reset_index().sort_values("detour")

    con = curve(con_all)
    ws  = curve(ws_df)

    print(f"\n(hole-crossing pairs sampled: {n_total})")
    print("=== upload-adequate: constrained (ours) ===")
    for _, r in con.iterrows():
        flag = "" if r['n'] >= 30 else "  (too few - ignore)"
        print(f"  B={r['knob']:>5.0f} : removed={r['removed']:5.1f}%  detour={r['detour']:5.1f}%  "
              f"(n={int(r['n'])}){flag}")
    print("=== upload-adequate: weighted-sum (baseline) ===")
    for _, r in ws.iterrows():
        print(f"  lam={r['knob']:>5.1f}: removed={r['removed']:5.1f}%  detour={r['detour']:5.1f}%  (n={int(r['n'])})")

    # ---- figure ----
    apply_style(usetex=True)
    fig, ax = plt.subplots()
    ax.plot(ws["detour"], ws["removed"], "-s", color=GREY, markersize=4,
            linewidth=1.4, label="Weighted-sum (baseline)", zorder=3)
    ax.plot(con["detour"], con["removed"], "-o", color=BLUE, markersize=4,
            linewidth=1.6, label="Constrained (ours)", zorder=4)
    ax.set_xlabel(r"Travel-time detour (\%)")
    ax.set_ylabel(r"Upload-inadequate distance removed (\%)")
    ax.legend(loc="lower right", fontsize=7)
    grid_box()
    save("upload_tradeoff", category="upload", fig=fig)
    plt.close(fig)

    # ---- save table ----
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/tables/upload_adequate_{stamp}.csv", index=False)
    print(f"\nsaved upload-adequate table + figure ({stamp})")