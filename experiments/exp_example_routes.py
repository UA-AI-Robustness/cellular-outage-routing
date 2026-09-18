"""Example-route figure: fastest route vs. budget-constrained connectivity route,
over the coverage map. House style, PDF+PNG, timestamped."""
from __future__ import annotations
import random
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types
from connroute.search.preferences import build_preferences, make_order
from connroute.search.lexico import lexico_route, constrained_route
from connroute.viz.style import apply_style, GREEN, RED, BLUE, save

MIN_OD_METERS = 2500.0
MIN_DEAD_REMOVED = 300.0      # fast route must cross a real hole
DEAD_BUDGET = 100.0          # metres of dead driving allowed on the connectivity route
N_EXAMPLES = 10              # <-- number of example figures


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


def plot_example(G, s, d, fast_path, conn_path, idx, meta):
    apply_style(usetex=False)
    dead = np.array([float(dta["dead_fraction"]) for _, _, dta in G.edges(data=True)])
    cmap = LinearSegmentedColormap.from_list("cov", [GREEN, "#F4D03F", RED])
    base_colors = [cmap(x) for x in dead]

    fig, ax = ox.plot_graph(
        G, edge_color=base_colors, edge_linewidth=0.5,
        node_size=0, bgcolor="white", show=False, close=False, figsize=(7, 7),
    )

    def draw(path, color, lw):
        xs = [G.nodes[n]["x"] for n in path]; ys = [G.nodes[n]["y"] for n in path]
        ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", zorder=5)

    draw(fast_path, BLUE, 2.6)
    draw(conn_path, "#1a1a1a", 2.2)
    ax.scatter([G.nodes[s]["x"]], [G.nodes[s]["y"]], c="black", s=40, zorder=6, marker="o")
    ax.scatter([G.nodes[d]["x"]], [G.nodes[d]["y"]], c="black", s=55, zorder=6, marker="*")

    handles = [Line2D([0], [0], color=BLUE, lw=2.6, label="Fastest"),
               Line2D([0], [0], color="#1a1a1a", lw=2.2, label="Connectivity (budget %dm)" % int(DEAD_BUDGET))]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)
    save(f"example_route_{idx}", category="routes", fig=fig)
    plt.close(fig)


if __name__ == "__main__":
    G, cfg = load_graph()
    prefs = build_preferences(cfg)
    base = make_order(prefs, ["time"])

    rng = random.Random(cfg.seed)
    nodes = list(G.nodes())
    found = 0
    tries = 0

    pbar = tqdm(total=N_EXAMPLES, desc="finding examples", unit="fig")
    while found < N_EXAMPLES and tries < 40000:
        tries += 1
        s, d = rng.choice(nodes), rng.choice(nodes)
        if s == d or straight_line_m(G, s, d) < MIN_OD_METERS:
            continue
        fast_path, _ = lexico_route(G, s, d, base)
        if not fast_path:
            continue
        fast_dead, fast_time = path_dead_time(G, fast_path)
        if fast_dead < MIN_DEAD_REMOVED:
            continue
        conn_path, ctime, cbud = constrained_route(G, s, d,
                                                   budget_attr="d_dead",
                                                   budget=DEAD_BUDGET,
                                                   cost_attr="time")
        if not conn_path or conn_path == fast_path:
            continue
        conn_dead, conn_time = path_dead_time(G, conn_path)
        detour = 100.0 * (conn_time - fast_time) / fast_time if fast_time else 0.0
        found += 1
        tqdm.write(f"example {found}: {s}->{d}  fast_dead={fast_dead:.0f}m -> "
                   f"conn_dead={conn_dead:.0f}m  detour={detour:.0f}%")
        plot_example(G, s, d, fast_path, conn_path, found, None)
        pbar.update(1)
    pbar.close()

    print(f"\nproduced {found} example figures in results/figures/routes/"
          if found else "no illustrative example found — loosen thresholds")