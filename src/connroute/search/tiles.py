"""Tile partition over the SF road graph (adapted from the ACM tile_partition).

Overlays an N x N grid on the projected (metric) graph and builds:
  tile_of[node]         -> (row, col)
  tiles[(row,col)]      -> [node ids]
  cross_edges[(Ta,Tb)]  -> [(u, v, k)] edges crossing tile Ta -> Tb

The heuristic (Module 1) uses cross_edges to build a coarse tile graph.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import osmnx as ox

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types


@dataclass
class TilePartition:
    n_tiles: int
    bounds: tuple            # (xmin, ymin, xmax, ymax)
    cell_w: float
    cell_h: float
    tile_of: dict            # node -> (row, col)
    tiles: dict              # (row, col) -> [nodes]
    cross_edges: dict        # (Ta, Tb) -> [(u, v, k)]

    @property
    def n_nonempty(self):
        return len(self.tiles)

    def summary(self):
        return (f"TilePartition({self.n_tiles}x{self.n_tiles}, "
                f"{self.n_nonempty} non-empty tiles, "
                f"{len(self.cross_edges)} directed tile-adjacencies)")


def build_tiles(G, n_tiles: int) -> TilePartition:
    # graph is already projected to metres (UTM) -> node 'x','y' are metric
    xs = [float(d["x"]) for _, d in G.nodes(data=True)]
    ys = [float(d["y"]) for _, d in G.nodes(data=True)]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cell_w = (xmax - xmin) / n_tiles or 1.0
    cell_h = (ymax - ymin) / n_tiles or 1.0

    def cell_of(x, y):
        col = min(max(int((x - xmin) / cell_w), 0), n_tiles - 1)
        row = min(max(int((y - ymin) / cell_h), 0), n_tiles - 1)
        return (row, col)

    tile_of = {}
    tiles = defaultdict(list)
    for n, d in G.nodes(data=True):
        rc = cell_of(float(d["x"]), float(d["y"]))
        tile_of[n] = rc
        tiles[rc].append(n)

    cross_edges = defaultdict(list)
    for u, v, k in G.edges(keys=True):
        ta, tb = tile_of[u], tile_of[v]
        if ta != tb:
            cross_edges[(ta, tb)].append((u, v, k))

    return TilePartition(
        n_tiles=n_tiles,
        bounds=(xmin, ymin, xmax, ymax),
        cell_w=cell_w, cell_h=cell_h,
        tile_of=tile_of,
        tiles=dict(tiles),
        cross_edges=dict(cross_edges),
    )


def _report(part: TilePartition):
    sizes = sorted(len(v) for v in part.tiles.values())
    if not sizes:
        return
    import statistics as st
    print(f"  nodes/tile: min={sizes[0]} median={int(st.median(sizes))} max={sizes[-1]}")
    total_cross = sum(len(v) for v in part.cross_edges.values())
    print(f"  total cross-edges: {total_cross}")


if __name__ == "__main__":
    import argparse
    cfg = load_config()
    G = ox.load_graphml(cache_path(cfg))
    _coerce_types(G)
    for _, nd in G.nodes(data=True):
        nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])

    ap = argparse.ArgumentParser(description="Build tile partition over the SF graph.")
    ap.add_argument("--tiles", type=int, default=16,
                    help="grid size N (means N x N tiles). e.g. --tiles 16")
    args = ap.parse_args()

    part = build_tiles(G, args.tiles)
    print(f"[--tiles {args.tiles}] {part.summary()}")
    _report(part)