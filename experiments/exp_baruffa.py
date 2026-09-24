"""Ours (constrained) vs Baruffa et al. (radio-discount scalarization) vs
k-shortest reranking baseline, plus shortest reference. Parallel."""
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.baruffa import compute_radio_weight
from connroute.experiment.parallel import run_parallel
from connroute.viz.style import apply_style, grid_box, BLUE, GREY, RED, save

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
    print(f"running {len(pairs)} pairs (ours vs Baruffa vs k-shortest) in parallel...")
    rows = run_parallel("experiments.workers:baruffa_pair", pairs, desc="pairs", n_workers=8)
    df = pd.DataFrame(rows)

    def curve(method):
        sub = df[df["method"] == method]
        agg = sub.groupby("knob").agg(
            detour=("detour", "median"),
            reduction=("reduction", "median"),
            n=("s", "count")).reset_index().sort_values("detour")
        return agg

    con = curve("constrained")
    bar = curve("baruffa")
    sho = df[df["method"] == "shortest"]
    ksh = df[df["method"] == "kshortest_k5"]          # <-- k-shortest rows

    print(f"\neligible pairs: {df.groupby(['s','d']).ngroups}")
    print("=== constrained (ours) ===")
    for _, r in con.iterrows():
        print(f"  B={r['knob']:>5.0f}: reduction={r['reduction']:5.1f}%  detour={r['detour']:5.1f}%  (n={int(r['n'])})")
    print("=== Baruffa (alpha) ===")
    for _, r in bar.iterrows():
        print(f"  a={r['knob']:>5.1f}: reduction={r['reduction']:5.1f}%  detour={r['detour']:5.1f}%  (n={int(r['n'])})")
    if len(sho):
        print(f"shortest: reduction={sho['reduction'].median():.1f}%  detour={sho['detour'].median():.1f}%  (n={len(sho)})")
    if len(ksh):
        print(f"k-shortest (k=5): reduction={ksh['reduction'].median():.1f}%  "
              f"detour={ksh['detour'].median():.1f}%  (n={len(ksh)})")
        print("k-shortest reduction dist: median={:.1f}  p75={:.1f}  p90={:.1f}  max={:.1f}".format(
            ksh["reduction"].median(), ksh["reduction"].quantile(0.75),
            ksh["reduction"].quantile(0.90), ksh["reduction"].max()))
    else:
        print("k-shortest: NO ROWS (worker not producing kshortest_k5)")

    # ---- figures ----
    apply_style(usetex=True)
    bar_by_alpha = curve("baruffa").sort_values("knob")
    bar_by_det   = bar.sort_values("detour")

    ksh_pt = (ksh["detour"].median(), ksh["reduction"].median()) if len(ksh) else None

    def base_axes():
        fig, ax = plt.subplots()
        ax.plot(con["detour"], con["reduction"], "-o", color=BLUE, markersize=4,
                linewidth=1.6, label="Ours", zorder=4)
        # if ksh_pt is not None:                         # <-- k-shortest marker
            # ax.scatter([ksh_pt[0]], [ksh_pt[1]], marker="D", color=RED, s=42,
            #            zorder=5, label="$k$-shortest rerank")
        ax.set_xlabel(r"Travel-time detour (\%)")
        ax.set_ylabel(r"Predicted exposure reduction (\%)")
        return fig, ax

    # --- Figure A: Baruffa as SCATTER ---
    figA, axA = base_axes()
    axA.scatter(bar_by_alpha["detour"], bar_by_alpha["reduction"],
                marker="s", color=GREY, s=28, zorder=3, label=r"Baruffa \emph{et al.}")
    axA.legend(loc="lower right", fontsize=7)
    grid_box()
    save("baruffa_tradeoff_scatter", category="baselines", fig=figA)
    plt.close(figA)

    # --- Figure B: Baruffa as LINE ---
    figB, axB = base_axes()
    axB.plot(bar_by_det["detour"], bar_by_det["reduction"], "-s", color=GREY,
             markersize=4, linewidth=1.4, label=r"Baruffa \emph{et al.}", zorder=3)
    axB.legend(loc="lower right", fontsize=7)
    grid_box()
    save("baruffa_tradeoff_line", category="baselines", fig=figB)
    plt.close(figB)

    # ---- save table ----
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/tables/baruffa_{stamp}.csv", index=False)
    print(f"\nsaved two figures + table ({stamp})")