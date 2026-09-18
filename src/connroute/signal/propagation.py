"""Stage 1b — propagation physics: distance -> path loss -> RSRP -> SINR.

Pure functions over NumPy arrays. No I/O, no graph. First-pass model:
line-of-sight only (single path-loss exponent); building-blockage added later.
All power quantities in dB/dBm; distances in metres.
"""
from __future__ import annotations
import numpy as np

from connroute.config import SignalCfg


def path_loss_db(d_m: np.ndarray, cfg: SignalCfg, nlos: np.ndarray | None = None) -> np.ndarray:
    """Log-distance path loss (dB).  PL = PL0 + 10 n log10(d/d0) [+ X if NLOS].

    d_m  : distances point->tower, metres (clipped at d0 to avoid log of <1 ratio).
    nlos : optional boolean array; where True, use n_nlos and add the NLOS penalty.
           If None (first-pass), everything is treated as LOS.
    """
    d = np.maximum(d_m, cfg.d0_m)                     # never below the reference distance
    if nlos is None:
        n = cfg.n_los
        x = 0.0
    else:
        n = np.where(nlos, cfg.n_nlos, cfg.n_los)
        x = np.where(nlos, cfg.nlos_penalty_db, 0.0)
    return cfg.pl0_db + 10.0 * n * np.log10(d / cfg.d0_m) + x


def rsrp_dbm(d_m: np.ndarray, cfg: SignalCfg, nlos: np.ndarray | None = None) -> np.ndarray:
    """Received power (dBm).  RSRP = P_tx + G_tower + G_device - PL."""
    pl = path_loss_db(d_m, cfg, nlos)
    return cfg.p_tx_dbm + cfg.g_tower_db + cfg.g_device_db - pl


def noise_floor_dbm(cfg: SignalCfg) -> float:
    """Thermal noise floor (dBm).  N0 = -174 + 10 log10(B) + NF."""
    return -174.0 + 10.0 * np.log10(cfg.bandwidth_hz) + cfg.noise_figure_db


def sinr_db(
    rsrp_serve_dbm: np.ndarray,
    cfg: SignalCfg,
    interference_dbm: np.ndarray | None = None,
) -> np.ndarray:
    """SINR (dB).

    First-pass (interference_dbm is None): noise-limited SNR = RSRP_serve - N0.
    Full case: 10 log10( P_serve / (N0 + sum P_other) ), all converted to linear.
    """
    n0 = noise_floor_dbm(cfg)
    if interference_dbm is None:
        return rsrp_serve_dbm - n0                    # dB subtraction = linear division
    # interference-limited: work in linear (milliwatts), then back to dB
    p_serve = 10.0 ** (rsrp_serve_dbm / 10.0)
    p_int = 10.0 ** (interference_dbm / 10.0)
    p_noise = 10.0 ** (n0 / 10.0)
    return 10.0 * np.log10(p_serve / (p_noise + p_int))


def dbm_to_mw(dbm: np.ndarray) -> np.ndarray:
    """Helper: dBm -> linear power in milliwatts (for summing interference)."""
    return 10.0 ** (np.asarray(dbm) / 10.0)


if __name__ == "__main__":
    # quick sanity check with the config defaults
    from connroute.config import load_config
    cfg = load_config().signal

    print(f"noise floor N0 = {noise_floor_dbm(cfg):.1f} dBm")
    for d in [50, 200, 800, 2000, 5000]:
        d_arr = np.array([float(d)])
        pl = path_loss_db(d_arr, cfg)[0]
        rsrp = rsrp_dbm(d_arr, cfg)[0]
        snr = sinr_db(np.array([rsrp]), cfg)[0]
        covered = "covered" if snr >= cfg.theta_db else "DEAD"
        print(f"  d={d:5d} m   PL={pl:6.1f} dB   RSRP={rsrp:7.1f} dBm   SNR={snr:6.1f} dB   [{covered}]")