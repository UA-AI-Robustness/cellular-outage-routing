"""Modeled (uniform) load vs traffic-informed load: does real traffic change
routing? Rate-inadequate routing under each load model, common pairs, restricted
to a common feasible population so the two trade-off curves are comparable."""
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
N_WORKERS = 4          # laptop-safe; set None on a big machine for all cores


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
    print(f"running {len(pairs)} pairs (modeled vs traffic load), n_workers={N_WORKERS}...")
    rows = run_parallel("experiments.workers:loadcompare_pair", pairs,
                        desc="pairs", n_workers=N_WORKERS)
    df = pd.DataFrame(rows)

    # ---- route divergence (headline finding) ----
    div = df[df["load"] == "diverge"]
    if len(div):
        print(f"\nroute divergence (modeled vs traffic, B=200): "
              f"{div['removed_pct'].mean():.0f}% of {len(div)} feasible pairs")

    # ---- common feasible population (both models, all budgets) ----
    def complete_pairs(tag):
        sub = df[df["load"] == tag]
        budgets = sorted(sub["knob"].unique())
        if not budgets:
            return set()
        cnt = sub.groupby(["s", "d"])["knob"].nunique()
        return set(cnt[cnt == len(budgets)].index)

    common = complete_pairs("model") & complete_pairs("traffic")
    print(f"common feasible pairs (both models, all budgets): {len(common)}")

    # if the common set is too small, fall back to looser budgets only
    LOOSE = {400, 800, 1600}
    use_loose = len(common) < 25
    if use_loose:
        print("  common set small -> restricting to looser budgets {400,800,1600}")

    def curve(tag):
        sub = df[df["load"] == tag].copy()
        if use_loose:
            sub = sub[sub["knob"].isin(LOOSE)]
            # common set over the looser budgets
            cnt = sub.groupby(["s", "d"])["knob"].nunique()
            keep = set(cnt[cnt == len(LOOSE & set(sub["knob"].unique()))].index)
            sub = sub[sub.apply(lambda r: (r["s"], r["d"]) in keep, axis=1)]
        else:
            sub = sub[sub.apply(lambda r: (r["s"], r["d"]) in common, axis=1)]
        return sub.groupby("knob").agg(
            detour=("detour_pct", "median"),
            removed=("removed_pct", "median"),
            n=("s", "count")).reset_index().sort_values("knob")

    for tag in ("model", "traffic"):
        c = curve(tag)
        print(f"\n=== {tag} load ===")
        for _, r in c.iterrows():
            print(f"  B={r['knob']:>5.0f}: removed={r['removed']:5.1f}%  "
                  f"detour={r['detour']:5.1f}%  (n={int(r['n'])})")

    # ---- figure ----
    apply_style(usetex=True)
    cm = curve("model").sort_values("detour")
    ctf = curve("traffic").sort_values("detour")
    fig, ax = plt.subplots()
    ax.plot(cm["detour"], cm["removed"], "-o", color=GREY, markersize=4,
            linewidth=1.5, label="Uniform load", zorder=3)
    ax.plot(ctf["detour"], ctf["removed"], "-s", color=BLUE, markersize=4,
            linewidth=1.5, label="Traffic-informed load", zorder=4)
    ax.set_xlabel(r"Travel-time detour (\%)")
    ax.set_ylabel(r"Rate-inadequate distance removed (\%)")
    ax.legend(loc="lower left", fontsize=7)
    grid_box()
    save("loadcompare_tradeoff", category="load", fig=fig)
    plt.close(fig)

    # ---- save table ----
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/tables/loadcompare_{stamp}.csv", index=False)
    print(f"\nsaved load-comparison table + figure ({stamp})")