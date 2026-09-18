"""Stage 2b — attach routing-objective attributes to the signal graph.

Reads the cached signal-layer graph (from Stage 1c), computes:
  - d_dead  : dead distance per edge  = dead_fraction * length
  - q        : per-user rate per edge  (load-aware; needs N(T))
  - q_dwell  : q * dwell               (= q * travel_time)
and casts GraphML string attrs to numeric. Returns a graph ready for the search.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import math
import numpy as np
import osmnx as ox

from connroute.config import Config, load_config
from connroute.signal.layer import cache_path as signal_cache_path


# edge attributes that must be numeric after loading from GraphML
_FLOAT_ATTRS = ("length", "travel_time", "speed_kph",
                "s_min", "s_mean", "dead_fraction")
_INT_ATTRS = ("serve", "n_samples")


def _coerce_types(Gp) -> None:
    """GraphML stores everything as strings; cast the attrs we use back to numbers."""
    for _, _, data in Gp.edges(data=True):
        for a in _FLOAT_ATTRS:
            if a in data:
                data[a] = float(data[a])
        for a in _INT_ATTRS:
            if a in data:
                data[a] = int(float(data[a]))


def _rate_from_sinr(s_mean_db: float, bandwidth_hz: float) -> float:
    """Uncongested Shannon rate (bits/s) from mean SINR (dB). R = B log2(1+lin(SINR))."""
    if not math.isfinite(s_mean_db):
        return 0.0
    lin = 10.0 ** (s_mean_db / 10.0)
    return bandwidth_hz * math.log2(1.0 + lin)


def attach_objectives(cfg: Config, Gp=None):
    """Load (or take) the signal graph and attach objective attributes to edges."""
    if Gp is None:
        Gp = ox.load_graphml(signal_cache_path(cfg))
    _coerce_types(Gp)

    scfg, lcfg = cfg.signal, cfg.load

    # --- 1. per-edge dead distance and raw rate ---
    for _, _, data in Gp.edges(data=True):
        length = data["length"]
        data["d_dead"] = data["dead_fraction"] * length                 # metres
        data["r_raw"] = _rate_from_sinr(data["s_mean"], scfg.bandwidth_hz)
        # 'time' alias for the time preference (seconds)
        data["time"] = data["travel_time"]
        # dwell = travel_time (seconds)
        data["dwell"] = data["travel_time"]

    # --- 2. tower load N(T): sum traffic over each tower's served edges ---
    # traffic proxy: use edge vehicle-count if present, else a flat 1 per edge.
    cell_traffic = defaultdict(float)
    for _, _, data in Gp.edges(data=True):
        T = data.get("serve", -1)
        if T < 0:
            continue
        traffic = float(data.get("traffic", 1.0))   # placeholder until TomTom flow wired in
        cell_traffic[T] += traffic

    def N_of(T: int) -> float:
        base = cell_traffic.get(T, 0.0) * lcfg.devices_per_vehicle * lcfg.operator_share
        return base + lcfg.background_devices

    # --- 3. per-user rate q(e) and data-volume contribution q_dwell ---
    # --- 3. per-user rate q(e), gated by coverage ---
    q_vals = []
    for _, _, data in Gp.edges(data=True):
        T = data.get("serve", -1)
        covered_frac = 1.0 - data["dead_fraction"]        # NEW: fraction of edge that's usable
        if T < 0 or data["r_raw"] <= 0.0 or covered_frac <= 0.0:
            q = 0.0
        else:
            q = (data["r_raw"] * covered_frac) / max(1.0, N_of(T))   # gate rate by coverage
        data["q_raw"] = q
        q_vals.append(q)

    # normalize q to [0,1] across the graph so it sits beside other prefs
    q_arr = np.array(q_vals)
    q_max = q_arr.max() if q_arr.size and q_arr.max() > 0 else 1.0
    for _, _, data in Gp.edges(data=True):
        data["q"] = data["q_raw"] / q_max
        data["q_dwell"] = data["q"] * data["dwell"]

    return Gp


def cache_path(cfg: Config) -> Path:
    safe = cfg.city.split(",")[0].strip().lower().replace(" ", "_")
    return Path(cfg.cache_dir) / "signal" / f"{safe}_objectives.graphml"


def summarize_objectives(Gp) -> None:
    d_dead = np.array([d["d_dead"] for _, _, d in Gp.edges(data=True)])
    q = np.array([d["q"] for _, _, d in Gp.edges(data=True)])
    print("\n=== objectives summary ===")
    print(f"d_dead (m):  total={d_dead.sum():,.0f}  mean={d_dead.mean():.1f}  max={d_dead.max():.1f}")
    print(f"q (norm):    min={q.min():.3f}  median={np.median(q):.3f}  max={q.max():.3f}")
    print(f"edges with q=0 (no service): {(q==0).mean()*100:.1f}%")


if __name__ == "__main__":
    cfg = load_config()
    Gp = attach_objectives(cfg)
    summarize_objectives(Gp)

    p = cache_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(Gp, p)
    print(f"\ncached objectives graph: {p}")