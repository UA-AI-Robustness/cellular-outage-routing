"""Stage 2b — attach routing-objective attributes to the signal graph.

Reads the cached signal-layer graph (from Stage 1c), computes:
  - d_dead   : dead distance per edge  = dead_fraction * length
  - q         : per-user rate per edge (modeled load; placeholder traffic)
  - q_tt      : per-user rate under traffic-informed load (TomTom), if enabled
  - q_dwell   : q * dwell               (= q * travel_time)
  - d_lowrate / d_lowupload : fractional below-threshold distances
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
                "s_min", "s_mean", "dead_fraction",
                "frac_below_cover", "frac_below_call", "frac_below_upload")
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


def _normalize_q(Gp, raw_attr, out_attr):
    """Normalize a raw per-edge rate to [0,1] by its 95th percentile (positive
    values), clipped at 1, and write it to out_attr."""
    vals = np.array([d[raw_attr] for _, _, d in Gp.edges(data=True)])
    pos = vals[vals > 0]
    ref = np.percentile(pos, 95) if pos.size else 1.0
    if ref <= 0:
        ref = 1.0
    for _, _, data in Gp.edges(data=True):
        data[out_attr] = min(data[raw_attr] / ref, 1.0)


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
        data["time"] = data["travel_time"]
        data["dwell"] = data["travel_time"]

    # --- 2. MODELED tower load N(T): flat traffic placeholder (existing) ---
    cell_traffic = defaultdict(float)
    for _, _, data in Gp.edges(data=True):
        T = data.get("serve", -1)
        if T < 0:
            continue
        traffic = float(data.get("traffic", 1.0))   # flat placeholder
        cell_traffic[T] += traffic

    def N_of(T: int) -> float:
        base = cell_traffic.get(T, 0.0) * lcfg.devices_per_vehicle * lcfg.operator_share
        return base + lcfg.background_devices

    # --- 3. modeled per-user rate q(e), gated by coverage ---
    for _, _, data in Gp.edges(data=True):
        T = data.get("serve", -1)
        covered_frac = 1.0 - data["dead_fraction"]
        if T < 0 or data["r_raw"] <= 0.0 or covered_frac <= 0.0:
            q = 0.0
        else:
            q = (data["r_raw"] * covered_frac) / max(1.0, N_of(T))
        data["q_raw"] = q
    _normalize_q(Gp, "q_raw", "q")
    for _, _, data in Gp.edges(data=True):
        data["q_dwell"] = data["q"] * data["dwell"]

    # --- 3b. OPTIONAL: traffic-informed load q_tt (TomTom), alongside modeled q ---
    use_traffic = bool(getattr(lcfg, "use_traffic", False))
    if use_traffic:
        from connroute.signal.traffic import build_traffic_grid, congestion_at
        from connroute.graph.build import cache_path as graph_cache_path
        Gll = ox.load_graphml(graph_cache_path(cfg))   # unprojected graph (lat/lon)
        grid = build_traffic_grid(Gll, cfg, grid_side=12)

        # per-tower traffic from congestion: congested edges (low rho) -> more vehicles
        cell_traffic_tt = defaultdict(float)
        for u, v, k, data in Gp.edges(keys=True, data=True):
            T = data.get("serve", -1)
            if T < 0:
                continue
            if u in Gll.nodes and v in Gll.nodes:
                la = 0.5 * (float(Gll.nodes[u]["y"]) + float(Gll.nodes[v]["y"]))
                lo = 0.5 * (float(Gll.nodes[u]["x"]) + float(Gll.nodes[v]["x"]))
                rho = congestion_at(grid, la, lo)
            else:
                rho = 1.0
            veh = float(data["length"]) * (1.0 / max(rho, 0.05))   # congested -> more vehicles
            cell_traffic_tt[T] += veh

        # normalize traffic so its mean ~1, comparable to the modeled placeholder scale
        if cell_traffic_tt:
            mean_tt = np.mean(list(cell_traffic_tt.values()))
            for T in list(cell_traffic_tt):
                cell_traffic_tt[T] /= max(mean_tt, 1e-9)

        def N_tt(T: int) -> float:
            base = cell_traffic_tt.get(T, 1.0) * lcfg.devices_per_vehicle * lcfg.operator_share
            return base + lcfg.background_devices

        for _, _, data in Gp.edges(data=True):
            T = data.get("serve", -1)
            covered_frac = 1.0 - data["dead_fraction"]
            if T < 0 or data["r_raw"] <= 0.0 or covered_frac <= 0.0:
                q = 0.0
            else:
                q = (data["r_raw"] * covered_frac) / max(1.0, N_tt(T))
            data["q_tt_raw"] = q
        _normalize_q(Gp, "q_tt_raw", "q_tt")
        for _, _, data in Gp.edges(data=True):
            data["q_tt_dwell"] = data["q_tt"] * data["dwell"]

    # --- 4. fractional below-threshold distances (point-level, like d_dead) ---
    # call/upload adequacy use per-point SINR-threshold fractions from the signal
    # layer (frac_below_call, frac_below_upload), consistent with continuity (d_dead).
    for _, _, data in Gp.edges(data=True):
        length = float(data["length"])
        data["d_lowrate"]   = float(data["frac_below_call"])   * length   # call-adequate
        data["d_lowupload"] = float(data["frac_below_upload"]) * length   # upload-adequate

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
    # traffic-informed q, if present
    if any("q_tt" in d for _, _, d in Gp.edges(data=True)):
        qtt = np.array([d.get("q_tt", np.nan) for _, _, d in Gp.edges(data=True)])
        qtt = qtt[~np.isnan(qtt)]
        print(f"q_tt (norm): min={qtt.min():.3f}  median={np.median(qtt):.3f}  max={qtt.max():.3f}  (traffic-informed)")


if __name__ == "__main__":
    cfg = load_config()
    Gp = attach_objectives(cfg)
    summarize_objectives(Gp)

    p = cache_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(Gp, p)
    print(f"\ncached objectives graph: {p}")