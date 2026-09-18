"""Parallel OD-pair evaluation across CPU cores.

The graph is loaded ONCE per worker process (via an initializer) so it isn't
re-serialized for every task. Each task is just an (s, d) pair; the worker runs
the routing and returns metric rows. No algorithm change -> identical results.
"""
from __future__ import annotations
import os
from multiprocessing import Pool
from tqdm import tqdm
import osmnx as ox

from connroute.config import load_config
from connroute.objectives.attach import cache_path, _coerce_types

# module-level globals populated in each worker by _init_worker
_G = None
_CFG = None
_WORKER_FN = None


def _init_worker(worker_fn_name_module):
    """Runs once per worker process: load graph + config into globals."""
    global _G, _CFG, _WORKER_FN
    _CFG = load_config()
    G = ox.load_graphml(cache_path(_CFG)); _coerce_types(G)
    for _, _, dta in G.edges(data=True):
        for a in ("d_dead", "q", "q_dwell", "time", "length"):
            dta[a] = float(dta[a])
    for _, nd in G.nodes(data=True):
        nd["x"] = float(nd["x"]); nd["y"] = float(nd["y"])
    _G = G
    # resolve the worker function from "module:function"
    import importlib
    mod_name, fn_name = worker_fn_name_module.split(":")
    _WORKER_FN = getattr(importlib.import_module(mod_name), fn_name)


def _run_task(task):
    """Called in a worker: task is whatever the worker_fn expects; we pass the graph."""
    return _WORKER_FN(_G, _CFG, task)


def run_parallel(worker_fn_path: str, tasks: list, n_workers: int | None = None,
                 desc: str = "tasks"):
    """Run worker_fn over tasks across processes.

    worker_fn_path : "module.path:function_name" — the per-task function, signature
                     fn(G, cfg, task) -> list_of_rows (or any picklable result).
    tasks          : list of picklable task objects (e.g. (s, d) tuples).
    n_workers      : defaults to os.cpu_count().
    Returns a flat list of all results (rows) concatenated.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 4
    results = []
    with Pool(processes=n_workers, initializer=_init_worker,
              initargs=(worker_fn_path,)) as pool:
        for res in tqdm(pool.imap_unordered(_run_task, tasks, chunksize=4),
                        total=len(tasks), desc=desc, unit="task"):
            if res:
                results.extend(res)
    return results