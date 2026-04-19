"""
utils/helpers.py
────────────────
Shared math utilities: normalization, smoothing, color mappings, caching.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import functools


# ──────────────────────────────────────────────────
# Colormaps
# ──────────────────────────────────────────────────

#: Neon alpha colormap: bearish (magenta) → neutral (deep blue) → bullish (green)
NEON_ALPHA_CMAP = LinearSegmentedColormap.from_list(
    "neon_alpha",
    [
        (0.000, "#ff0066"),  # -1.0  neon magenta (bearish)
        (0.200, "#8800cc"),  # -0.6  vivid violet
        (0.350, "#2200aa"),  # -0.3  deep indigo
        (0.500, "#0d1a6e"),  #  0.0  deep navy neutral
        (0.650, "#0066cc"),  # +0.3  electric blue
        (0.800, "#00e5ff"),  # +0.6  electric cyan
        (1.000, "#39ff14"),  # +1.0  neon green (bullish)
    ],
)

#: Energy surface colormap: dark → indigo → cyan  (intensity map, always positive)
NEON_ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "neon_energy",
    [
        (0.000, "#050510"),  # 0.0  background black
        (0.200, "#0d0d3a"),  # 0.2  deep navy
        (0.400, "#1a0a6e"),  # 0.4  indigo
        (0.600, "#0044cc"),  # 0.6  blue
        (0.800, "#00ccff"),  # 0.8  cyan
        (1.000, "#ffffff"),  #  1.0 white-hot peak
    ],
)


def alpha_to_rgba(alpha_arr: np.ndarray, cmap=NEON_ALPHA_CMAP) -> np.ndarray:
    """
    Map alpha values in [-1, 1] → RGBA array in [0,1].

    Parameters
    ----------
    alpha_arr : array-like
        Values in [-1, 1].
    cmap : matplotlib colormap
        Defaults to NEON_ALPHA_CMAP.

    Returns
    -------
    np.ndarray  shape (*alpha_arr.shape, 4)
    """
    arr = np.asarray(alpha_arr, dtype=float)
    normalized = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    return cmap(normalized)  # (..., 4)


def energy_to_rgba(energy_arr: np.ndarray, cmap=NEON_ENERGY_CMAP) -> np.ndarray:
    """Map energy values in [0, 1] → RGBA array."""
    arr = np.clip(np.asarray(energy_arr, dtype=float), 0.0, 1.0)
    return cmap(arr)


# ──────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────

def minmax_normalize(arr: np.ndarray, axis: int = 0, eps: float = 1e-9) -> np.ndarray:
    """
    Min-max normalize along *axis* → [0, 1].
    NaN values are preserved.
    """
    arr = np.asarray(arr, dtype=float)
    mn = np.nanmin(arr, axis=axis, keepdims=True)
    mx = np.nanmax(arr, axis=axis, keepdims=True)
    return (arr - mn) / (mx - mn + eps)


def zscore_normalize(arr: np.ndarray, axis: int = 0, eps: float = 1e-9) -> np.ndarray:
    """Z-score normalize along *axis*. NaN-safe."""
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr, axis=axis, keepdims=True)
    sigma = np.nanstd(arr, axis=axis, keepdims=True)
    return (arr - mu) / (sigma + eps)


# ──────────────────────────────────────────────────
# Smoothing
# ──────────────────────────────────────────────────

def gaussian_smooth_2d(arr: np.ndarray, sigma_time: float = 3.0,
                        sigma_asset: float = 0.5) -> np.ndarray:
    """
    Apply Gaussian smoothing over both time (axis=0) and asset (axis=1) dimensions.
    Used to create wave-like surface transitions between frames.

    Parameters
    ----------
    arr         : (T, A) float array
    sigma_time  : smoothing in the time dimension (days)
    sigma_asset : smoothing across asset dimension
    """
    out = gaussian_filter1d(arr, sigma=sigma_time, axis=0, mode="nearest")
    if sigma_asset > 0:
        out = gaussian_filter1d(out, sigma=sigma_asset, axis=1, mode="nearest")
    return out


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple centered rolling mean, NaN at edges."""
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        start = max(0, i - window // 2)
        end = min(len(arr), i + window // 2 + 1)
        out[i] = np.nanmean(arr[start:end])
    return out


# ──────────────────────────────────────────────────
# Interpolation helpers
# ──────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b at fraction t."""
    return a + (b - a) * float(t)


def smooth_step(t: float) -> float:
    """Smoothstep curve: 3t²-2t³, used for eased transitions."""
    t = np.clip(float(t), 0, 1)
    return t * t * (3 - 2 * t)


# ──────────────────────────────────────────────────
# Simple cache decorator (in-process)
# ──────────────────────────────────────────────────

def memoize(fn):
    """Thin memoize wrapper for expensive compute functions."""
    return functools.lru_cache(maxsize=None)(fn)
