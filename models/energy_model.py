"""
models/energy_model.py
───────────────────────
Compute the 4D market energy field E(t, a) from pre-computed features.

  E(t, a) = w1·volatility + w2·|momentum_raw| + w3·volume_spike + w4·sentiment

The energy surface is:
  - Gaussian-smoothed in both time and asset dimensions
  - Used as Z-values on the 3D surface
  - Color-coded by the alpha directional signal α(t,a)
  - Anomalies flagged for glow markers and shock beams
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from utils.helpers import minmax_normalize, gaussian_smooth_2d


# ── Default weight configuration ────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "volatility":    0.35,
    "momentum":      0.25,   # uses |momentum_raw| for energy, signed for alpha
    "volume_spike":  0.25,
    "sentiment":     0.15,
}


def compute_energy(
    features: dict,
    weights: dict | None = None,
    smooth_sigma_time: float = 3.0,
    smooth_sigma_asset: float = 0.6,
    anomaly_threshold_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the market energy surface and derived signals.

    Parameters
    ----------
    features          : dict from compute_features()
    weights           : override DEFAULT_WEIGHTS
    smooth_sigma_time : Gaussian σ along time axis (days)
    smooth_sigma_asset: Gaussian σ along asset axis
    anomaly_threshold_sigma : z-score threshold for anomaly detection

    Returns
    -------
    E       : np.ndarray (T, A)  energy surface, values in [0, 1]
    alpha   : np.ndarray (T, A)  directional signal in [-1, 1]
              > 0 = bullish (neon green), < 0 = bearish (neon magenta)
    anomaly : np.ndarray (T, A)  bool, True where |z-score(E)| > threshold
    """
    print("⚡  Computing energy surface …")

    if weights is None:
        weights = DEFAULT_WEIGHTS

    # ── Weighted sum ────────────────────────────────────────────────────
    vol_arr   = features["volatility"].values           # (T, A)
    mom_raw   = features["momentum_raw"].values         # signed
    vspike    = features["volume_spike"].values
    sent      = features["sentiment"].values

    # Absolute normalised momentum for energy magnitude
    mom_abs = np.abs(minmax_normalize(mom_raw, axis=0))

    E_raw = (
        weights.get("volatility",   0.35) * vol_arr
        + weights.get("momentum",   0.25) * mom_abs
        + weights.get("volume_spike", 0.25) * vspike
        + weights.get("sentiment",  0.15) * sent
    )

    # ── Gaussian smoothing for wave-like motion ─────────────────────────
    E_smooth = gaussian_smooth_2d(E_raw,
                                   sigma_time=smooth_sigma_time,
                                   sigma_asset=smooth_sigma_asset)

    # Final normalise to [0, 1]
    E = minmax_normalize(E_smooth, axis=0)

    # ── Alpha directional signal ────────────────────────────────────────
    # α = sign(momentum) × tanh(4 × |normalised momentum|)  → [-1, 1]
    sign = np.sign(mom_raw)
    mom_norm = minmax_normalize(np.abs(mom_raw), axis=0)
    alpha = sign * np.tanh(4.0 * mom_norm)
    # Light smoothing on alpha too (cosmetic continuity)
    alpha = gaussian_smooth_2d(alpha, sigma_time=2.0, sigma_asset=0.4)
    alpha = np.clip(alpha, -1.0, 1.0)

    # ── Anomaly detection ───────────────────────────────────────────────
    E_mean = E.mean(axis=0, keepdims=True)
    E_std  = E.std(axis=0, keepdims=True) + 1e-9
    z_score = (E - E_mean) / E_std
    anomaly = np.abs(z_score) > anomaly_threshold_sigma

    print(f"   Energy shape:  {E.shape}  min={E.min():.3f}  max={E.max():.3f}")
    print(f"   Alpha range:   [{alpha.min():.3f}, {alpha.max():.3f}]")
    print(f"   Anomaly count: {anomaly.sum()} ({100*anomaly.mean():.1f}%)")
    return E, alpha, anomaly


def identify_shocks(
    price_df: pd.DataFrame,
    log_return_threshold_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    Identify major price shock events (crashes and pumps).

    Returns
    -------
    pd.DataFrame with columns ['date', 'asset', 'log_return', 'direction']
    where direction = 'crash' | 'pump'
    """
    log_ret = np.log(price_df / price_df.shift(1))
    mu = log_ret.mean()
    sigma = log_ret.std()

    records = []
    for col in log_ret.columns:
        threshold = log_return_threshold_sigma * sigma[col]
        mask = np.abs(log_ret[col]) > threshold
        for date, val in log_ret[col][mask].items():
            records.append({
                "date":       date,
                "asset":      col,
                "log_return": val,
                "direction":  "pump" if val > 0 else "crash",
            })

    shocks = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"   Shock events detected: {len(shocks)}")
    return shocks
