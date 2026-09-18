"""Live-call trade-off + baseline: minimize call-inadequate distance (d_lowrate).
Our constrained method (sweep budget) vs weighted-sum baseline (sweep lambda),
mirroring the continuity/baseline comparison. Parallel."""
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
    print(f"running {len(pairs)} pairs (live-call) in parallel...")
    rows = run_parallel("experiments.workers:livecall_pair", pairs, desc="pairs")
    df = pd.DataFrame(rows)

    # ---- fix the population: keep only pairs feasible at ALL constrained budgets ----
    con_all = df[df["method"] == "constrained"]
    budgets = sorted(con_all["knob"].unique())
    # a pair is "complete" if it has a constrained result for every budget
    counts = con_all.groupby(["s", "d"])["knob"].nunique()
    complete_pairs = set(counts[counts == len(budgets)].index)
    print(f"pairs feasible at all {len(budgets)} budgets: {len(complete_pairs)} "
          f"(of {con_all.groupby(['s','d']).ngroups} with any constrained result)")

    def _is_complete(row):
        return (row["s"], row["d"]) in complete_pairs

    # constrained curve: fixed population (complete pairs only) -> monotonic
    con_df = con_all[con_all.apply(_is_complete, axis=1)]
    # baseline curve: already a constant population (always feasible), keep as-is
    ws_df = df[df["method"] == "weighted_sum"]

    def curve(sub):
        agg = sub.groupby("knob").agg(
            detour=("detour_pct", "median"),
            removed=("lowrate_removed_pct", "median"),
            n=("s", "count")).reset_index()
        return agg.sort_values("detour")

    con = curve(con_df)
    ws = curve(ws_df)


    def curve(method):
        sub = df[df["method"] == method]
        agg = sub.groupby("knob").agg(
            detour=("detour_pct", "median"),
            removed=("lowrate_removed_pct", "median"),
            n=("s", "count")).reset_index()
        return agg.sort_values("detour")

    # con = curve("constrained")
    # ws = curve("weighted_sum")

    print("\n=== live-call: constrained (ours) ===")
    for _, r in con.iterrows():
        print(f"  B={r['knob']:>5.0f} : removed={r['removed']:5.1f}%  detour={r['detour']:5.1f}%  (n={int(r['n'])})")
    print("=== live-call: weighted-sum (baseline) ===")
    for _, r in ws.iterrows():
        print(f"  lam={r['knob']:>5.1f}: removed={r['removed']:5.1f}%  detour={r['detour']:5.1f}%  (n={int(r['n'])})")

    apply_style(usetex=True)
    fig, ax = plt.subplots()
    ax.plot(ws["detour"], ws["removed"], "-s", color=GREY, markersize=4,
            linewidth=1.4, label="Weighted-sum (baseline)", zorder=3)
    ax.plot(con["detour"], con["removed"], "-o", color=BLUE, markersize=4,
            linewidth=1.6, label="Constrained (ours)", zorder=4)
    ax.set_xlabel(r"Travel-time detour (\%)")
    ax.set_ylabel(r"Call-inadequate distance removed (\%)")
    ax.legend(loc="lower right", fontsize=7)
    grid_box()
    save("livecall_tradeoff", category="livecall", fig=fig)
    plt.close(fig)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/tables/livecall_{stamp}.csv", index=False)
    print(f"\nsaved live-call table + figure ({stamp})")