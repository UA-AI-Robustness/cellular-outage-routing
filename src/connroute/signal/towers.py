"""Stage 1a — load OpenCelliD towers, filter to a city bbox, enable fast lookup."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

# OpenCelliD CSV column order (the export has NO header row)
OCID_COLUMNS = [
    "radio", "mcc", "net", "area", "cell", "unit",
    "lon", "lat", "range", "samples", "changeable",
    "created", "updated", "averageSignal",
]

# San Francisco bounding box (lat/lon); a little margin beyond the city edge
SF_BBOX = {"lat_min": 37.70, "lat_max": 37.83, "lon_min": -122.52, "lon_max": -122.35}

# Project WGS84 -> a metric CRS (UTM zone 10N covers SF) so distances are in metres
_TO_METRIC = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)


@dataclass
class TowerSet:
    """Towers within the bbox, with a KD-tree over projected (x, y) in metres."""
    df: pd.DataFrame          # columns: radio, lon, lat, x, y
    tree: cKDTree
    xy: np.ndarray            # (N, 2) projected coords, metres

    def near(self, lat: float, lon: float, radius_m: float) -> np.ndarray:
        """Row indices of towers within radius_m of a (lat, lon) point."""
        x, y = _TO_METRIC.transform(lon, lat)   # note: xy order = (lon, lat)
        return np.array(self.tree.query_ball_point([x, y], r=radius_m), dtype=int)


def load_towers(
    path: str | Path,
    bbox: dict = SF_BBOX,
    radios: tuple[str, ...] | None = ("LTE",),   # None = keep all technologies
) -> TowerSet:
    """Load the OpenCelliD CSV, filter to bbox (+ optional radio), build a KD-tree."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"tower file not found: {path.resolve()}")

    # read only the columns we need; file has no header, so pass names + usecols by index
    df = pd.read_csv(
        path,
        header=None,
        names=OCID_COLUMNS,
        usecols=["radio", "lon", "lat"],
        dtype={"radio": "category", "lon": "float64", "lat": "float64"},
    )

    n_total = len(df)

    # bbox filter (this is what turns all-US into just-SF)
    m = (
        df["lat"].between(bbox["lat_min"], bbox["lat_max"])
        & df["lon"].between(bbox["lon_min"], bbox["lon_max"])
    )
    df = df[m].copy()
    n_bbox = len(df)

    # optional technology filter
    if radios is not None:
        df = df[df["radio"].isin(radios)].copy()
    n_final = len(df)

    if n_final == 0:
        raise ValueError("no towers left after filtering — check bbox / radio settings")

    # project to metres and build KD-tree for fast radius queries
    x, y = _TO_METRIC.transform(df["lon"].values, df["lat"].values)
    df["x"], df["y"] = x, y
    df = df.reset_index(drop=True)
    xy = np.column_stack([df["x"].values, df["y"].values])
    tree = cKDTree(xy)

    print(f"towers: {n_total:,} total  ->  {n_bbox:,} in bbox  ->  {n_final:,} after radio filter")
    return TowerSet(df=df[["radio", "lon", "lat", "x", "y"]], tree=tree, xy=xy)


if __name__ == "__main__":
    ts = load_towers("data/raw/opencellid_us_310.csv")
    print(ts.df.head())
    print(f"\nSF tower count: {len(ts.df):,}")