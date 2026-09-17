"""Stage 0 — build the urban road graph with length, speed, and travel time."""
from __future__ import annotations
from pathlib import Path
import osmnx as ox

from connroute.config import Config, load_config


def build_graph(cfg: Config):
    """Download the drivable road network and attach edge attributes.

    Returns a networkx MultiDiGraph. Every edge (u, v, k) carries:
      - length      : segment length in metres
      - speed_kph   : free-flow speed (imputed from OSM tags by road type)
      - travel_time : seconds to traverse the edge  (this is dwell(e) later)
    """
    G = ox.graph_from_place(cfg.city, network_type=cfg.network_type)
    G = ox.distance.add_edge_lengths(G)      # length (m)
    G = ox.add_edge_speeds(G)                # speed_kph
    G = ox.add_edge_travel_times(G)          # travel_time (s)
    return G


def cache_path(cfg: Config) -> Path:
    safe = cfg.city.split(",")[0].strip().lower().replace(" ", "_")
    return Path(cfg.cache_dir) / "graphs" / f"{safe}.graphml"


def build_or_load(cfg: Config, use_cache: bool = True):
    """Load the graph from cache if present, else build and cache it."""
    p = cache_path(cfg)
    if use_cache and p.exists():
        print(f"loading cached graph: {p}")
        return ox.load_graphml(p)
    G = build_graph(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, p)
    print(f"built and cached graph: {p}")
    return G


def summarize(G) -> None:
    u, v, k, data = next(iter(G.edges(keys=True, data=True)))
    print(f"nodes: {G.number_of_nodes():,}   edges: {G.number_of_edges():,}")
    print("sample edge attributes:")
    for key in ("length", "speed_kph", "travel_time", "name", "highway"):
        print(f"   {key:12s}: {data.get(key)}")


if __name__ == "__main__":
    cfg = load_config()
    G = build_or_load(cfg)
    summarize(G)