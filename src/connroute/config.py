"""Load config.yaml into a typed, attribute-access object."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class SignalCfg:
    sample_spacing_m: float
    tower_radius_m: float
    pl0_db: float
    d0_m: float
    n_los: float
    n_nlos: float
    nlos_penalty_db: float
    p_tx_dbm: float
    g_tower_db: float
    g_device_db: float
    bandwidth_hz: float
    noise_figure_db: float
    theta_db: float


@dataclass
class LoadCfg:
    devices_per_vehicle: float
    operator_share: float
    background_devices: float


@dataclass
class Config:
    city: str
    network_type: str
    seed: int
    cache_dir: str
    signal: SignalCfg
    load: LoadCfg


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path.resolve()}")
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        city=raw["city"],
        network_type=raw["network_type"],
        seed=raw.get("seed", 42),
        cache_dir=raw.get("cache_dir", "cache"),
        signal=SignalCfg(**raw["signal"]),
        load=LoadCfg(**raw["load"]),
    )