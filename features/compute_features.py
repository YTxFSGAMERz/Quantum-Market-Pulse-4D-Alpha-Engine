"""
features/compute_features.py
─────────────────────────────
Compute per-asset financial features from price + volume DataFrames.

All output arrays are min-max normalized to [0, 1] for consistent
Z-axis scaling on the 3D energy surface.

Returns a dict keyed by feature name, each value is a
pd.DataFrame of shape (T, A) aligned to the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from utils.helpers import minmax_normalize


# ── Wilder RSI ─────────────────────────────────────────────────────────────

def _wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's RSI rescaled to [0, 1]."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 / (1.0 + rs)           # inverted: 0=overbought, 100=oversold
    return (rsi / 100.0).rename(series.name)  # → [0, 1]


# ── Main computation ────────────────────────────────────────────────────────

def compute_features(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    vol_window: int = 21,
    mom_window: int = 50,
    rsi_window: int = 14,
    vol_spike_window: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    Compute all financial features for each asset.

    Parameters
    ----------
    price_df   : (T, A) daily closing prices
    volume_df  : (T, A) daily volumes
    vol_window : rolling std window for volatility (days)
    mom_window : SMA window for momentum (days)
    rsi_window : Wilder RSI period (days)
    vol_spike_window : rolling mean window for volume spike ratio

    Returns
    -------
    Dict with keys:
        log_return    (T, A)  [normalised to 0-1]
        volatility    (T, A)  [normalised to 0-1]
        momentum      (T, A)  [normalised to 0-1 from signed -∞..+∞]
        rsi           (T, A)  [already 0-1]
        volume_spike  (T, A)  [normalised to 0-1, clamped at 5×]
        sentiment     (T, A)  [normalised to 0-1]
    """
    print("⚙️   Computing features …")

    feats: dict[str, pd.DataFrame] = {}
    n_assets = price_df.shape[1]

    # ── Log returns ─────────────────────────────────────────────────────
    log_ret = np.log(price_df / price_df.shift(1))   # NaN on first row
    feats["log_return"] = pd.DataFrame(
        minmax_normalize(log_ret.values, axis=0),
        index=price_df.index,
        columns=price_df.columns,
    )

    # ── Rolling volatility (annualised ÷ √252 for daily → keep raw) ─────
    vol = log_ret.rolling(vol_window, min_periods=5).std()
    feats["volatility"] = pd.DataFrame(
        minmax_normalize(vol.values, axis=0),
        index=price_df.index,
        columns=price_df.columns,
    )

    # ── Momentum: price / SMA(50) − 1  (signed) ────────────────────────
    sma = price_df.rolling(mom_window, min_periods=5).mean()
    mom = price_df / sma - 1.0
    feats["momentum"] = pd.DataFrame(
        minmax_normalize(mom.values, axis=0),
        index=price_df.index,
        columns=price_df.columns,
    )
    # Keep raw signed momentum separately for alpha signal direction
    feats["momentum_raw"] = mom

    # ── Wilder RSI ───────────────────────────────────────────────────────
    rsi_dict = {col: _wilder_rsi(price_df[col], rsi_window) for col in price_df.columns}
    feats["rsi"] = pd.DataFrame(rsi_dict)

    # ── Volume spike: ratio to rolling mean, clamped at 5× ─────────────
    vol_mean = volume_df.rolling(vol_spike_window, min_periods=5).mean().replace(0, np.nan)
    spike = (volume_df / vol_mean).clip(upper=5.0).fillna(1.0)
    feats["volume_spike"] = pd.DataFrame(
        minmax_normalize(spike.values, axis=0),
        index=price_df.index,
        columns=price_df.columns,
    )

    # ── Synthetic sentiment ─────────────────────────────────────────────
    # Modelled as a tanh-transformed cumulative sum of (log_return × vol)
    # Seeded identically so it's reproducible and differs per asset.
    sentiment_cols = {}
    for i, col in enumerate(price_df.columns):
        rng = np.random.default_rng(2024 + i)
        eps = rng.standard_normal(len(price_df)) * 0.02
        raw_vol = vol[col].bfill().ffill().values
        raw_log = log_ret[col].fillna(0).values
        sent = np.tanh(np.cumsum(raw_log * raw_vol * 30 + eps))
        sentiment_cols[col] = sent
    sent_df = pd.DataFrame(sentiment_cols, index=price_df.index)
    feats["sentiment"] = pd.DataFrame(
        minmax_normalize(sent_df.values, axis=0),
        index=price_df.index,
        columns=price_df.columns,
    )

    # ── Fill remaining NaNs with column medians ─────────────────────────
    for key in feats:
        if key == "momentum_raw":
            feats[key] = feats[key].fillna(0.0)
        else:
            feats[key] = feats[key].apply(lambda s: s.fillna(s.median()))

    print(f"   Features: {[k for k in feats if k != 'momentum_raw']}")
    print(f"   Shape: {feats['volatility'].shape}")
    return feats
