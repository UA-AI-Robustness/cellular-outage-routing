"""Stage 2a — Preference definitions: the objects the mixed-aggregation search consumes.

Each Preference bundles:
  - name
  - op        : how contributions combine along a route ('sum' or 'min')
  - identity  : neutral starting value for that operator
  - sentinel  : 'worst' value (unvisited)
  - direction : 'min' or 'max' (do we want the aggregated value small or large?)
  - contrib   : edge-attr dict -> per-edge contribution (float)

We keep everything in a single MAXIMIZE convention internally: a minimized
objective is stored as its negation, so 'better' always means 'larger'.
The direction field records the user-facing intent; contrib already returns the
sign-corrected value so the engine only ever maximizes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math

from connroute.config import Config


@dataclass(frozen=True)
class Preference:
    name: str
    op: str                       # 'sum' or 'min'
    identity: float               # neutral element for op (maximize convention)
    sentinel: float               # worst value (unvisited)
    contrib: Callable[[dict], float]   # edge attributes -> per-edge contribution
    direction: str                # 'min' or 'max' (user intent, for reporting)

    def combine(self, acc: float, c: float) -> float:
        if self.op == "sum":
            return acc + c
        if self.op == "min":
            return min(acc, c)
        raise ValueError(f"unknown op: {self.op}")


# ---- operator constants (maximize convention) ----
# sum : identity 0, worst -inf
# min : identity +inf, worst -inf
_SUM_ID, _MIN_ID, _WORST = 0.0, math.inf, -math.inf


# ---- edge-contribution functions ----
# Edges carry precomputed attributes from Stage 2b (attach.py):
#   d_dead   : dead distance on the edge (metres)          [minimize -> negate]
#   length   : segment length (metres)                     [minimize -> negate]
#   time     : travel time (seconds)                       [minimize -> negate]
#   q        : per-user rate on the edge (normalized 0..1)  [maximize]
#   q_dwell  : q * dwell  (data-volume contribution)        [maximize]

def _neg(attr: str) -> Callable[[dict], float]:
    return lambda e: -float(e[attr])

def _pos(attr: str) -> Callable[[dict], float]:
    return lambda e: float(e[attr])


def build_preferences(cfg: Config) -> dict[str, Preference]:
    """Return the catalog of available preferences, keyed by name."""
    return {
        # continuity: minimize dead distance (additive) -> store -d_dead, maximize
        "dead_exposure": Preference(
            name="dead_exposure", op="sum", identity=_SUM_ID, sentinel=_WORST,
            contrib=_neg("d_dead"), direction="min",
        ),
        # travel time: minimize seconds (additive) -> store -time, maximize
        "time": Preference(
            name="time", op="sum", identity=_SUM_ID, sentinel=_WORST,
            contrib=_neg("time"), direction="min",
        ),
        # distance: minimize metres (additive) -> store -length, maximize
        "distance": Preference(
            name="distance", op="sum", identity=_SUM_ID, sentinel=_WORST,
            contrib=_neg("length"), direction="min",
        ),
        # live-call: maximize the worst-served segment (bottleneck / min)
        "live_call": Preference(
            name="live_call", op="min", identity=_MIN_ID, sentinel=_WORST,
            contrib=_pos("q"), direction="max",
        ),
        # upload: maximize delivered data = sum of q*dwell (additive)
        "upload": Preference(
            name="upload", op="sum", identity=_SUM_ID, sentinel=_WORST,
            contrib=_pos("q_dwell"), direction="max",
        ),
    }


def make_order(prefs: dict[str, Preference], names: list[str]) -> list[Preference]:
    """Build a priority-ordered list of Preference objects from names (highest first)."""
    missing = [n for n in names if n not in prefs]
    if missing:
        raise KeyError(f"unknown preferences: {missing}")
    return [prefs[n] for n in names]