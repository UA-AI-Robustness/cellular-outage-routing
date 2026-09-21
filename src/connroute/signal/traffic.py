"""TomTom traffic-flow grid query for load-informed QoS.
Queries the TomTom Flow Segment Data API over an N x N grid covering the
graph's bounding box, returns a congestion field (grid cell -> congestion
ratio in (0,1], where lower = more congested). Cached to disk so the API is
queried once, respecting the free-tier quota.

Set the key in the environment (never in code):
    conda env config vars set TOMTOM_API_KEY="..."
"""
from __future__ import annotations
import os, json, time
from pathlib import Path
import numpy as np
import requests

FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"


def _cache_path(cfg, grid_side):
    Path("cache/traffic").mkdir(parents=True, exist_ok=True)
    safe = cfg.city.split(",")[0].strip().lower().replace(" ", "_")
    return Path("cache/traffic") / f"{safe}_grid{grid_side}.json"


def build_traffic_grid(G, cfg, grid_side=12, pause=0.05, force=False):
    """Query TomTom over a grid_side x grid_side grid on the graph's lat/lon
    bbox. Returns dict: {'lats':[...], 'lons':[...], 'rho':[[...]]} where rho is
    the congestion ratio (current/free-flow speed) per grid cell in (0,1].
    Cached to disk; set force=True to re-query."""
    cache = _cache_path(cfg, grid_side)
    if cache.exists() and not force:
        print(f"[traffic] loading cached grid: {cache}")
        return json.loads(cache.read_text())

    key = os.environ.get("TOMTOM_API_KEY", "")
    if not key:
        raise SystemExit("TOMTOM_API_KEY not set in environment")

    # bbox from graph nodes (unprojected lat/lon)
    lats = [float(nd["y"]) for _, nd in G.nodes(data=True)]
    lons = [float(nd["x"]) for _, nd in G.nodes(data=True)]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    grid_lats = np.linspace(lat_min, lat_max, grid_side)
    grid_lons = np.linspace(lon_min, lon_max, grid_side)

    rho = [[1.0] * grid_side for _ in range(grid_side)]   # default: free-flow (1.0)
    n_ok = n_fail = 0
    print(f"[traffic] querying TomTom {grid_side}x{grid_side} = {grid_side**2} points...")
    for i, la in enumerate(grid_lats):
        for j, lo in enumerate(grid_lons):
            try:
                r = requests.get(FLOW_URL, params={"point": f"{la},{lo}", "key": key},
                                 timeout=15)
                if r.status_code == 200:
                    d = r.json()["flowSegmentData"]
                    cur = float(d.get("currentSpeed", 0))
                    free = float(d.get("freeFlowSpeed", 1)) or 1.0
                    rho[i][j] = max(min(cur / free, 1.0), 0.05)   # clip to (0.05, 1]
                    n_ok += 1
                else:
                    n_fail += 1   # keep default 1.0 (no congestion info)
            except Exception:
                n_fail += 1
            time.sleep(pause)   # be gentle on the API
    print(f"[traffic] ok={n_ok} fail={n_fail}")

    out = {"lats": grid_lats.tolist(), "lons": grid_lons.tolist(), "rho": rho,
           "bbox": [lat_min, lon_min, lat_max, lon_max]}
    cache.write_text(json.dumps(out))
    print(f"[traffic] cached -> {cache}")
    return out


def congestion_at(grid, lat, lon):
    """Nearest-grid-cell congestion ratio rho in (0,1] for a point."""
    lats = grid["lats"]; lons = grid["lons"]; rho = grid["rho"]
    i = int(np.argmin([abs(lat - gl) for gl in lats]))
    j = int(np.argmin([abs(lon - gl) for gl in lons]))
    return rho[i][j]