"""Stage 1c — build the per-edge coverage map.

For every edge: sample points every Delta metres, find nearby towers, compute
RSRP per (point, tower), pick the serving tower, get its SINR, label the point
covered/dead. Reduce each edge to: S_min, S_mean, serve(e), dead_fraction.
First-pass model: line-of-sight only (no building blockage yet).
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import networkx as nx
import osmnx as ox

from connroute.config import Config, load_config
from connroute.signal.towers import load_towers, TowerSet
from connroute.signal import propagation as prop


UTM_SF = "EPSG:32610"   # metric CRS for San Francisco (same as towers.py)


def _sample_points_xy(geom, spacing_m: float) -> np.ndarray:
    """Return (M, 2) array of (x, y) points along a projected LineString, every ~spacing_m.

    Always includes both endpoints; guarantees at least 2 points.
    """
    length = geom.length
    if length <= spacing_m:
        pts = [geom.interpolate(0.0), geom.interpolate(length)]
    else:
        n = int(np.floor(length / spacing_m)) + 1        # number of intervals
        dists = np.linspace(0.0, length, n + 1)
        pts = [geom.interpolate(d) for d in dists]
    return np.array([(p.x, p.y) for p in pts])


def build_signal_layer(cfg: Config, towers: TowerSet, G, verbose: bool = True):
    """Enrich a graph with per-edge signal summaries. Returns the projected graph."""
    scfg = cfg.signal
    n0 = prop.noise_floor_dbm(scfg)

    # project the graph to metres so edge geometries and tower coords share a CRS
    if verbose:
        print("projecting graph to UTM (metres)...")
    Gp = ox.project_graph(G, to_crs=UTM_SF)

    tower_xy = towers.xy                 # (N, 2) projected tower coords, metres
    radius = scfg.tower_radius_m
    theta = scfg.theta_db

    n_edges = Gp.number_of_edges()
    t0 = time.time()
    done = 0
    n_no_tower_points = 0

    for u, v, k, data in Gp.edges(keys=True, data=True):
        # get the edge geometry; simple edges may lack a 'geometry' -> build from node coords
        geom = data.get("geometry")
        if geom is None:
            x1, y1 = Gp.nodes[u]["x"], Gp.nodes[u]["y"]
            x2, y2 = Gp.nodes[v]["x"], Gp.nodes[v]["y"]
            from shapely.geometry import LineString
            geom = LineString([(x1, y1), (x2, y2)])

        pts = _sample_points_xy(geom, scfg.sample_spacing_m)   # (M, 2)

        # for each sample point: find nearby towers, compute best SINR
        point_sinr = np.full(len(pts), -np.inf)
        point_serve = np.full(len(pts), -1, dtype=int)

        # batch tower lookup: query_ball_point accepts multiple points at once
        neighbor_lists = towers.tree.query_ball_point(pts, r=radius)

        for i, (px, py) in enumerate(pts):
            idx = neighbor_lists[i]
            if len(idx) == 0:
                n_no_tower_points += 1
                continue                              # no tower in range -> stays -inf (dead)
            idx = np.asarray(idx, dtype=int)
            dx = tower_xy[idx, 0] - px
            dy = tower_xy[idx, 1] - py
            d = np.sqrt(dx * dx + dy * dy)            # distances to candidate towers, metres
            rsrp = prop.rsrp_dbm(d, scfg)             # LOS-only first pass (nlos=None)
            best = int(np.argmax(rsrp))
            point_serve[i] = idx[best]
            point_sinr[i] = rsrp[best] - n0           # noise-limited SNR (dB)

        covered = point_sinr >= theta
        dead_fraction = 1.0 - covered.mean()

        # dominant serving tower over covered points (fall back to all points)
        served = point_serve[point_serve >= 0]
        if served.size:
            vals, counts = np.unique(served, return_counts=True)
            serve_edge = int(vals[np.argmax(counts)])
        else:
            serve_edge = -1

        # finite SINR values only, for the summaries
        finite = point_sinr[np.isfinite(point_sinr)]
        s_min = float(finite.min()) if finite.size else -np.inf
        s_mean = float(finite.mean()) if finite.size else -np.inf

        data["s_min"] = s_min
        data["s_mean"] = s_mean
        data["serve"] = serve_edge
        data["dead_fraction"] = float(dead_fraction)
        data["n_samples"] = int(len(pts))

        done += 1
        if verbose and done % 5000 == 0:
            dt = time.time() - t0
            print(f"  {done:,}/{n_edges:,} edges  ({dt:.1f}s)")

    if verbose:
        dt = time.time() - t0
        print(f"done: {n_edges:,} edges in {dt:.1f}s")
        print(f"  sample points with no tower in range: {n_no_tower_points:,}")
    return Gp


def cache_path(cfg: Config) -> Path:
    safe = cfg.city.split(",")[0].strip().lower().replace(" ", "_")
    return Path(cfg.cache_dir) / "signal" / f"{safe}_signal.graphml"


def summarize_coverage(Gp) -> None:
    dead_fracs = np.array([d["dead_fraction"] for _, _, d in Gp.edges(data=True)])
    s_means = np.array([d["s_mean"] for _, _, d in Gp.edges(data=True)])
    s_means = s_means[np.isfinite(s_means)]
    fully_dead = (dead_fracs >= 0.999).mean()
    fully_cov = (dead_fracs <= 0.001).mean()
    print("\n=== coverage summary ===")
    print(f"edges fully covered : {fully_cov*100:5.1f}%")
    print(f"edges fully dead    : {fully_dead*100:5.1f}%")
    print(f"edges partially dead: {(1-fully_cov-fully_dead)*100:5.1f}%")
    print(f"mean dead fraction  : {dead_fracs.mean():.3f}")
    print(f"S_mean (dB): min={s_means.min():.1f}  median={np.median(s_means):.1f}  max={s_means.max():.1f}")


if __name__ == "__main__":
    cfg = load_config()
    towers = load_towers("data/raw/opencellid_us_310.csv")
    from connroute.graph.build import build_or_load
    G = build_or_load(cfg)

    Gp = build_signal_layer(cfg, towers, G)
    summarize_coverage(Gp)

    p = cache_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(Gp, p)
    print(f"\ncached signal layer: {p}")