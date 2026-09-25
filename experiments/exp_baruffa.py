"""Ours (constrained) vs Baruffa et al. (radio-discount scalarization, all
four radio-weight kinds) vs k-shortest reranking baseline, plus shortest
reference. Parallel."""
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
from connroute.viz.style import apply_style, grid_box, BLUE, GREY, RED, PALETTE, save

N_PAIRS = 500
MIN_OD_METERS = 2000.0
KINDS = ("on_off", "amplitude", "capacity", "tent")


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
    print(f"running {len(pairs)} pairs (ours vs Baruffa x4 kinds vs k-shortest) in parallel...")
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
    sho = df[df["method"] == "shortest"]
    ksh = df[df["method"] == "kshortest_k5"]

    bar_curves = {kind: curve(f"baruffa_{kind}") for kind in KINDS}

    print(f"\neligible pairs: {df.groupby(['s','d']).ngroups}")
    print("=== constrained (ours) ===")
    for _, r in con.iterrows():
        print(f"  B={r['knob']:>5.0f}: reduction={r['reduction']:5.1f}%  detour={r['detour']:5.1f}%  (n={int(r['n'])})")

    for kind in KINDS:
        bar = bar_curves[kind]
        print(f"=== Baruffa ({kind}, alpha) ===")
        if bar.empty:
            print("  NO ROWS")
            continue
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

    # ---- 2x2 subplot figures: Ours vs each Baruffa kind individually ----
    apply_style(usetex=True)
    kind_labels = {"on_off": "on--off", "amplitude": "amplitude",
                   "capacity": "capacity", "tent": "tent"}

    ksh_pt = (ksh["detour"].median(), ksh["reduction"].median()) if len(ksh) else None

    def make_grid(with_kshortest: bool, filename: str):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0), sharex=False, sharey=True)
        for ax, kind in zip(axes.flat, KINDS):
            bar = bar_curves[kind].sort_values("detour")
            ax.plot(con["detour"], con["reduction"], "-o", color=BLUE,
                    markersize=3.5, linewidth=1.5, label="Ours", zorder=4)
            if not bar.empty:
                ax.plot(bar["detour"], bar["reduction"], "-s", color=GREY,
                        markersize=3.5, linewidth=1.2,
                        label=r"Baruffa \emph{et al.}", zorder=3)
            if with_kshortest and ksh_pt is not None:
                ax.scatter([ksh_pt[0]], [ksh_pt[1]], marker="D", color=RED,
                          s=36, zorder=5, label="$k$-shortest")
            ax.set_title(kind_labels[kind], fontsize=9)
            ax.grid(True, linewidth=0.4, alpha=0.6)
            ax.set_xlabel(r"Detour (\%)", fontsize=8)

        axes[0, 0].set_ylabel(r"Exposure reduction (\%)", fontsize=8)
        axes[1, 0].set_ylabel(r"Exposure reduction (\%)", fontsize=8)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7,
                  bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        save(filename, category="baselines", fig=fig)
        plt.close(fig)

    make_grid(with_kshortest=True,  filename="baruffa_grid_with_kshortest")
    make_grid(with_kshortest=False, filename="baruffa_grid_no_kshortest")

    # ---- save table ----
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/tables/baruffa_{stamp}.csv", index=False)
    print(f"\nsaved two grid figures + table ({stamp})")