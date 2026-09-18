"""Headline result — the connectivity/detour trade-off curve.

Sweep the dead-exposure budget B over many OD pairs. For each B, aggregate the
median dead-exposure removed (connectivity gain) vs. the median detour (cost).
Plots the trade-off curve and saves the underlying table.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from tqdm import tqdm

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route, constrained_route
from connroute.viz.style import apply_style, grid_box, BLUE, RED, save

# ---- experiment parameters ----
N_PAIRS = 200
MIN_OD_METERS = 2000.0
MIN_FAST_DEAD = 100.0                       # only pairs whose fast route crosses a hole
BUDGETS = [25, 50, 100, 200, 400, 800]      # dead-exposure budgets (metres)
MAX_TRIES = 30000


def load_graph():
    cfg = load_config()
    G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
    for _, _, dta in G.edges(data=True):
        for a in ("d_dead", "q", "q_dwell", "time", "length"):
            dta[a] = float(dta[a])
    for _, nd in G.nodes(data=True):
        nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])
    return G, cfg


def straight_line_m(G, s, d):
    return float(np.hypot(G.nodes[s]["x"] - G.nodes[d]["x"],
                          G.nodes[s]["y"] - G.nodes[d]["y"]))


def path_dead_time(G, path):
    dd = tt = 0.0
    for a, b in zip(path[:-1], path[1:]):
        ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
        dd += float(ed["d_dead"]); tt += float(ed["time"])
    return dd, tt


def sample_hole_pairs(G, base, n_pairs, seed):
    """Sample OD pairs whose FASTEST route crosses a real hole (dead > MIN_FAST_DEAD)."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    pairs, tries = [], 0
    pbar = tqdm(total=n_pairs, desc="sampling hole pairs", unit="pair")
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
        pairs.append((s, d, fd, ft))
        pbar.update(1)
    pbar.close()
    return pairs


if __name__ == "__main__":
    G, cfg = load_graph()
    prefs = build_preferences(cfg)
    base = make_order(prefs, ["time"])

    pairs = sample_hole_pairs(G, base, N_PAIRS, cfg.seed)
    print(f"using {len(pairs)} hole-crossing pairs")

    rows = []
    for (s, d, fast_dead, fast_time) in tqdm(pairs, desc="pairs", unit="pair"):
        for B in BUDGETS:
            cp, ctime, cbud = constrained_route(G, s, d, budget_attr="d_dead",
                                                budget=float(B), cost_attr="time")
            if not cp:
                continue
            cdead, ctime2 = path_dead_time(G, cp)
            rows.append({
                "s": s, "d": d, "budget": B,
                "fast_dead": fast_dead, "fast_time": fast_time,
                "conn_dead": cdead, "conn_time": ctime2,
                "dead_removed": fast_dead - cdead,
                "dead_removed_pct": 100.0 * (fast_dead - cdead) / fast_dead if fast_dead else 0.0,
                "detour_pct": 100.0 * (ctime2 - fast_time) / fast_time if fast_time else 0.0,
            })

    df = pd.DataFrame(rows)

    # ---- aggregate: median gain and cost per budget ----
    agg = df.groupby("budget").agg(
        detour_med=("detour_pct", "median"),
        detour_q1=("detour_pct", lambda x: np.percentile(x, 25)),
        detour_q3=("detour_pct", lambda x: np.percentile(x, 75)),
        dead_removed_pct_med=("dead_removed_pct", "median"),
    ).reset_index()

    print("\n=== trade-off summary (median across pairs) ===")
    print(f"{'budget(m)':>10} {'dead removed %':>15} {'detour %':>10}")
    for _, r in agg.iterrows():
        print(f"{r['budget']:>10.0f} {r['dead_removed_pct_med']:>15.1f} {r['detour_med']:>10.1f}")

    # ---- save table ----
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/tables/tradeoff_raw_{stamp}.csv", index=False)
    agg.to_csv(f"results/tables/tradeoff_agg_{stamp}.csv", index=False)

    # ---- figure: detour (cost) vs dead-removed (gain) ----
    apply_style(usetex=True)
    fig, ax = plt.subplots()
    ax.plot(agg["detour_med"], agg["dead_removed_pct_med"],
            "-o", color=BLUE, markersize=5, linewidth=1.5, zorder=3)
    # annotate each point with its budget
    for _, r in agg.iterrows():
        ax.annotate(f"{int(r['budget'])}m",
                    (r["detour_med"], r["dead_removed_pct_med"]),
                    textcoords="offset points", xytext=(5, -8), fontsize=7)
    ax.set_xlabel(r"Travel-time detour (\%)")
    ax.set_ylabel(r"Dead-zone exposure removed (\%)")
    grid_box()
    save("tradeoff_curve", category="tradeoff", fig=fig)
    plt.close(fig)

    print(f"\nsaved tradeoff table + figure ({stamp})")