"""
data/fetch_data.py
──────────────────
Load BTC daily data from the bundled CSV and optionally hydrate a
multi-asset DataFrame (ETH, SPY, GOLD) from yfinance.

If yfinance is unavailable or network access fails, each sibling asset is
synthesised from a seeded random walk correlated to BTC log-returns, so
the renderer always works completely offline.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "btc-usd-max.csv"

# Animation slice: 2017 onset of major crypto era → yields ~9 years of rich data
START_DATE = "2017-01-01"

ASSETS = ["BTC", "ETH", "SPY", "GOLD"]

# Synthetic correlation parameters: (corr_with_btc, own_vol_scale)
_SYNTH_PARAMS: dict[str, tuple[float, float]] = {
    "ETH":  (0.80, 1.35),   # high corr, even more volatile
    "SPY":  (0.12, 0.40),   # low corr, much lower vol
    "GOLD": (-0.04, 0.25),  # near-zero corr, low vol
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_btc_csv(path: Path) -> pd.Series:
    """Parse the bundled btc-usd-max.csv → daily price Series."""
    df = pd.read_csv(path, parse_dates=["snapped_at"])
    df = df.rename(columns={"snapped_at": "date", "price": "close"})
    df = df.set_index("date").sort_index()
    df.index = df.index.tz_localize(None)          # strip UTC tz
    # Keep only the close price; forward-fill any gap days
    series = df["close"].resample("D").last().ffill()
    return series


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.Series | None:
    """Try yfinance; return None on any failure."""
    try:
        import yfinance as yf  # type: ignore
        raw = yf.download(ticker, start=start, end=end,
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None
        col = "Close" if "Close" in raw.columns else raw.columns[0]
        series = raw[col].squeeze().rename(ticker)
        series.index = pd.DatetimeIndex(series.index).tz_localize(None)
        return series.resample("D").last().ffill()
    except Exception:
        return None


def _synthesise_asset(btc_log_ret: pd.Series, corr: float,
                       vol_scale: float, seed: int) -> pd.Series:
    """
    Build a synthetic price series using correlated Brownian motion.

    Parameters
    ----------
    btc_log_ret  : BTC daily log returns (normalised to 1-day std ≈ 1)
    corr         : correlation coefficient with BTC
    vol_scale    : relative volatility vs BTC
    seed         : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    T = len(btc_log_ret)
    btc_std = btc_log_ret.std()

    # Correlated + idiosyncratic noise
    noise = rng.standard_normal(T)
    ret = corr * btc_log_ret.values + np.sqrt(1 - corr**2) * noise * btc_std
    ret = ret * vol_scale

    # Reconstruct price starting at 100
    price = 100.0 * np.exp(np.cumsum(ret))
    return pd.Series(price, index=btc_log_ret.index)


# ── Public API ─────────────────────────────────────────────────────────────

def load_data(
    start_date: str = START_DATE,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return two aligned DataFrames:

    Returns
    -------
    price_df : pd.DataFrame  shape (T, 4)
        Daily closing prices, columns = ['BTC','ETH','SPY','GOLD']
    volume_df : pd.DataFrame  shape (T, 4)
        Daily volume (BTC real, others synthetic or yfinance).
    """
    print("📥  Loading BTC data from CSV …")
    btc_price = _parse_btc_csv(CSV_PATH)
    btc_volume = _parse_btc_csv._get_volume(CSV_PATH)  # type: ignore[attr-defined]

    # Apply date slice
    if end_date is None:
        end_date = str(btc_price.index[-1].date())
    btc_price = btc_price.loc[start_date:end_date]
    btc_volume = btc_volume.loc[start_date:end_date]

    btc_log_ret = np.log(btc_price / btc_price.shift(1)).dropna()

    prices: dict[str, pd.Series] = {"BTC": btc_price}
    volumes: dict[str, pd.Series] = {"BTC": btc_volume}

    yf_map = {"ETH": "ETH-USD", "SPY": "SPY", "GOLD": "GC=F"}

    for i, asset in enumerate(["ETH", "SPY", "GOLD"]):
        print(f"   Fetching {asset} via yfinance …", end=" ", flush=True)
        series = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            series = _fetch_yfinance(yf_map[asset], start_date, end_date)

        if series is not None and len(series) > 50:
            print("✓ live")
            prices[asset] = series
            volumes[asset] = series * 0  # placeholder volume (not on yf raw for now)
        else:
            print("✗ synthesising …")
            corr, vol_scale = _SYNTH_PARAMS[asset]
            prices[asset] = _synthesise_asset(btc_log_ret, corr, vol_scale, seed=42 + i)
            volumes[asset] = pd.Series(
                np.abs(np.random.default_rng(99 + i).standard_normal(len(btc_price))) * 1e9,
                index=btc_price.index,
            )

    # Align to common calendar (BTC drives the index)
    price_df = pd.DataFrame(prices).reindex(btc_price.index).ffill().bfill()
    volume_df = pd.DataFrame(volumes).reindex(btc_price.index).ffill().bfill().fillna(0)

    print(f"✅  Loaded {len(price_df)} trading days  ({price_df.index[0].date()} → {price_df.index[-1].date()})")
    print(f"   Assets: {list(price_df.columns)}")
    return price_df, volume_df


# ── Monkey-patch helper ────────────────────────────────────────────────────
def _get_volume(path: Path) -> pd.Series:
    """Extract total_volume column from CSV."""
    df = pd.read_csv(path, parse_dates=["snapped_at"])
    df = df.rename(columns={"snapped_at": "date", "total_volume": "volume"})
    df = df.set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    series = df["volume"].resample("D").last().ffill().fillna(0)
    return series

_parse_btc_csv._get_volume = _get_volume  # type: ignore[attr-defined]
