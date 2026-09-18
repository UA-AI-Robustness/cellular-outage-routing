"""Divergence experiment — the make-or-break result.

For many OD pairs, run each priority order through the engine and measure how
often a connectivity-first order CHANGES the route vs. fastest, and by how much
(detour cost, dead-exposure reduction, bottleneck gain). Writes a CSV and a figure.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import pandas as pd
import osmnx as ox

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route, readable_scores
from tqdm import tqdm

# ---- experiment parameters ----
N_PAIRS = 300
MIN_OD_METERS = 2000.0     # straight-line minimum separation (non-trivial pairs)
MAX_TRIES = 20000


def load_graph():
    cfg = load_config()
    G = ox.load_graphml(cache_path(cfg)); _coerce_types(G)
    for _, _, dta in G.edges(data=True):
        for a in ("d_dead", "q", "q_dwell", "time", "length"):
            dta[a] = float(dta[a])
    # node coords as floats (for straight-line distance)
    for _, nd in G.nodes(data=True):
        nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])
    return G, cfg


def straight_line_m(G, s, d):
    xs, ys = G.nodes[s]["x"], G.nodes[s]["y"]
    xd, yd = G.nodes[d]["x"], G.nodes[d]["y"]
    return float(np.hypot(xs - xd, ys - yd))   # graph is projected -> metres


def sample_pairs(G, n_pairs, min_m, seed):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    pairs, tries = [], 0
    while len(pairs) < n_pairs and tries < MAX_TRIES:
        tries += 1
        s, d = rng.choice(nodes), rng.choice(nodes)
        if s == d or straight_line_m(G, s, d) < min_m:
            continue
        pairs.append((s, d))
    return pairs


def route_metrics(G, path):
    """Sum up interpretable metrics along a node path (using best parallel edge by time)."""
    d_dead = t = length = 0.0
    q_min = np.inf
    for a, b in zip(path[:-1], path[1:]):
        # pick the edge actually used: the min-time parallel (matches how fastest routes resolve)
        ed = min(G[a][b].values(), key=lambda e: float(e["time"]))
        d_dead += float(ed["d_dead"]); t += float(ed["time"]); length += float(ed["length"])
        q_min = min(q_min, float(ed["q"]))
    return {"dead_m": d_dead, "time_s": t, "length_m": length,
            "q_min": (0.0 if not np.isfinite(q_min) else q_min)}


def routes_differ(p1, p2):
    return p1 != p2


if __name__ == "__main__":
    G, cfg = load_graph()
    prefs = build_preferences(cfg)

    base = make_order(prefs, ["time"])                          # fastest route
    orders = {
        "dead_exposure": make_order(prefs, ["dead_exposure", "time"]),
        "live_call":     make_order(prefs, ["live_call", "time"]),
        "upload":        make_order(prefs, ["time", "upload"]),
    }

    pairs = sample_pairs(G, N_PAIRS, MIN_OD_METERS, cfg.seed)
    print(f"sampled {len(pairs)} OD pairs (>= {MIN_OD_METERS:.0f} m apart)")

    rows = []
    for i, (s, d) in enumerate(tqdm(pairs, desc="OD pairs", unit="pair"), 1):
        base_path, _ = lexico_route(G, s, d, base)
        if not base_path:
            continue
        bm = route_metrics(G, base_path)

        for name, order in orders.items():
            path, g = lexico_route(G, s, d, order)
            if not path:
                continue
            m = route_metrics(G, path)
            rows.append({
                "s": s, "d": d, "pref": name,
                "diverged": routes_differ(path, base_path),
                # cost: extra travel time vs fastest (%)
                "detour_pct": 100.0 * (m["time_s"] - bm["time_s"]) / bm["time_s"] if bm["time_s"] else 0.0,
                # gains, per preference type:
                "dead_reduction_m": bm["dead_m"] - m["dead_m"],       # positive = less dead distance
                "qmin_gain": m["q_min"] - bm["q_min"],                # positive = better worst-link
                "base_dead_m": bm["dead_m"],
            })
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)} pairs")

    df = pd.DataFrame(rows)

    # ---- aggregate summary ----
    print("\n=== divergence summary ===")
    for name in orders:
        sub = df[df["pref"] == name]
        frac = sub["diverged"].mean() * 100
        det = sub.loc[sub["diverged"], "detour_pct"].median()
        print(f"{name:14s}: diverged {frac:5.1f}%   median detour (when diverged) {det:5.1f}%")

    # among pairs whose fastest route DOES cross a dead zone, how often does dead_exposure help?
    dsub = df[(df["pref"] == "dead_exposure") & (df["base_dead_m"] > 1.0)]
    if len(dsub):
        print(f"\ndead_exposure, on pairs whose fast route crosses a hole ({len(dsub)} pairs):")
        print(f"  diverged {dsub['diverged'].mean()*100:.1f}%   "
              f"median dead-distance removed {dsub.loc[dsub['diverged'],'dead_reduction_m'].median():.0f} m   "
              f"median detour {dsub.loc[dsub['diverged'],'detour_pct'].median():.1f}%")

    # ---- save CSV ----
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = f"results/tables/divergence_{stamp}.csv"
    df.to_csv(csv, index=False)
    print(f"\nsaved {csv}")