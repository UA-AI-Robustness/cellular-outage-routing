"""Parallelized divergence/trade-off over many OD pairs. Same results as the
serial version, run across all CPU cores."""
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import pandas as pd
import osmnx as ox

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.experiment.parallel import run_parallel

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
        sl = np.hypot(G.nodes[s]["x"] - G.nodes[d]["x"], G.nodes[s]["y"] - G.nodes[d]["y"])
        if sl >= MIN_OD_METERS:
            pairs.append((s, d))
    return pairs


if __name__ == "__main__":
    pairs = sample_pairs()
    print(f"running {len(pairs)} pairs in parallel...")
    rows = run_parallel("experiments.workers:divergence_pair", pairs, desc="pairs")
    df = pd.DataFrame(rows)

    print("\n=== trade-off (median over pairs whose fast route crosses a hole) ===")
    hole = df[df["fast_dead"] > 100]
    for B in sorted(df["budget"].unique()):
        sub = hole[hole["budget"] == B]
        print(f"  B={B:4d}m : removed={sub['dead_removed_pct'].median():5.1f}%  "
              f"detour={sub['detour_pct'].median():5.1f}%  (n={len(sub)})")

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/tables/divergence_parallel_{stamp}.csv", index=False)
    print(f"\nsaved results/tables/divergence_parallel_{stamp}.csv")