"""
visualization/surface_renderer.py
──────────────────────────────────
Draw the neon 3D energy surface for a single video frame.

Called each frame with the 3D axes cleared (ax.cla()) beforehand.
Renders:
  1. Main energy surface  (plot_surface with RGBA facecolors)
  2. Floor contour projection
  3. Glow anomaly markers  (layered scatter for bloom effect)
  4. Shock event beams     (vertical lines at crash/pump dates)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from utils.helpers import alpha_to_rgba, NEON_ENERGY_CMAP, NEON_ALPHA_CMAP
from visualization.scene import BG_COLOR, TEXT_COLOR, DIM_GRID


# ── Constants ────────────────────────────────────────────────────────────────
ASSET_LABELS = ["BTC", "ETH", "SPY", "GOLD"]
WINDOW       = 90      # days shown on 3D surface at one time


# ── Neon blend: combine energy (Z-brightness) with alpha (color direction) ───

def _build_facecolors(E_win: np.ndarray, alpha_win: np.ndarray) -> np.ndarray:
    """
    Build per-vertex RGBA array blending energy intensity with alpha direction.

    Shape: (A, T, 4)  where A=assets, T=time window

    Strategy:
      - Base color comes from NEON_ALPHA_CMAP (direction)
      - Brightness is modulated by E value (high E = brighter/more saturated)
      - Very low E areas fade toward the background color
    """
    # alpha_win: (T, A)  → transpose to (A, T)
    alpha_T = alpha_win.T   # (A, T)
    E_T     = E_win.T       # (A, T)

    # Direction color from alpha
    rgba_dir = alpha_to_rgba(alpha_T, NEON_ALPHA_CMAP)  # (A, T, 4)

    # Brightness from energy: dim background regions, pop high-energy  
    brightness = 0.15 + 0.85 * E_T  # [0.15, 1.0]

    # Modulate RGB channels by brightness; keep alpha channel
    rgba = rgba_dir.copy()
    rgba[..., :3] = np.clip(rgba_dir[..., :3] * brightness[..., np.newaxis], 0, 1)

    # Edge alpha: slightly transparent to avoid harsh borders
    rgba[..., 3] = 0.90

    return rgba


# ── Main draw function ────────────────────────────────────────────────────────

def draw_surface(
    ax,
    E: np.ndarray,
    alpha_arr: np.ndarray,
    anomaly: np.ndarray,
    dates: pd.DatetimeIndex,
    t_idx: int,
    shocks: pd.DataFrame | None = None,
    frame_idx: int = 0,
    total_frames: int = 7200,
    window: int = WINDOW,
) -> None:
    """
    Draw the full 3D energy surface for the current frame.

    Parameters
    ----------
    ax         : Matplotlib Axes3D (already cleared via ax.cla())
    E          : (T, A) energy array
    alpha_arr  : (T, A) alpha directional signal [-1, 1]
    anomaly    : (T, A) bool  anomaly mask
    dates      : DatetimeIndex  of length T
    t_idx      : current frame's data index into E/alpha/anomaly
    shocks     : DataFrame from identify_shocks()  (optional)
    frame_idx  : current rendering frame number
    total_frames : total frames in video
    window     : number of days shown on surface
    """
    n_assets = E.shape[1]
    n_labels = min(n_assets, len(ASSET_LABELS))

    # Window slice: show [t_idx-window, t_idx]
    t_start = max(0, t_idx - window)
    t_end   = t_idx + 1
    actual_window = t_end - t_start

    E_win     = E[t_start:t_end, :]         # (W, A)
    alpha_win = alpha_arr[t_start:t_end, :] # (W, A)
    anom_win  = anomaly[t_start:t_end, :]   # (W, A)

    # ── Grid coordinates ──────────────────────────────────────────────────
    x = np.arange(actual_window)
    y = np.arange(n_assets)
    X, Y = np.meshgrid(x, y)   # (A, W)
    Z = E_win.T                  # (A, W)

    # ── Build face colors ─────────────────────────────────────────────────
    facecolors = _build_facecolors(E_win, alpha_win)  # (A, W, 4)

    # ── Surface ───────────────────────────────────────────────────────────
    ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        rstride=1, cstride=1,
        shade=True,
        lightsource=LightSource(azdeg=225, altdeg=45),
        linewidth=0.0,
        antialiased=True,
        alpha=0.92,
    )

    # ── Floor contour projection ──────────────────────────────────────────
    Z_floor = Z.min() - 0.05
    try:
        ax.contourf(
            X, Y, Z,
            zdir="z", offset=Z_floor,
            levels=8,
            cmap=NEON_ALPHA_CMAP,
            alpha=0.25,
        )
    except Exception:
        pass  # contour can fail on degenerate data; skip gracefully

    # ── Glow anomaly markers ──────────────────────────────────────────────
    anom_x, anom_y = np.where(anom_win.T)  # (A, W) indices
    if len(anom_x) > 0:
        anom_z    = Z[anom_x, anom_y]
        anom_alpha = alpha_win.T[anom_x, anom_y]

        # Outer glow halo (large, translucent)
        ax.scatter(
            anom_y, anom_x, anom_z,   # x=time, y=asset, z=energy
            s=120, c=anom_alpha,
            cmap=NEON_ALPHA_CMAP, vmin=-1, vmax=1,
            alpha=0.20, zorder=5, depthshade=False,
        )
        # Inner bright core
        ax.scatter(
            anom_y, anom_x, anom_z,
            s=25, c=anom_alpha,
            cmap=NEON_ALPHA_CMAP, vmin=-1, vmax=1,
            alpha=0.90, zorder=6, depthshade=False,
            edgecolors="#ffffff", linewidths=0.3,
        )

    # ── Shock beams ───────────────────────────────────────────────────────
    if shocks is not None and len(shocks) > 0:
        _draw_shock_beams(ax, shocks, dates, t_start, t_end,
                           Z_floor, Z.max(), frame_idx, total_frames)

    # ── Axis labels & style ───────────────────────────────────────────────
    _style_3d_axes(ax, dates[t_start:t_end], n_labels)


def _draw_shock_beams(ax, shocks, dates, t_start, t_end, z_floor, z_top,
                       frame_idx, total_frames):
    """Draw animated vertical beams at shock event dates visible in current window."""
    z_range = z_top - z_floor
    if z_range < 1e-6:
        return

    window_dates = dates[t_start:t_end]

    for _, row in shocks.iterrows():
        if row["date"] not in window_dates:
            continue

        t_pos = window_dates.get_loc(row["date"])
        a_pos = ASSET_LABELS.index(row["asset"]) if row["asset"] in ASSET_LABELS else 0

        color = "#00ff88" if row["direction"] == "pump" else "#ff2200"

        # Animated fade: beams are bright just after they appear, then dim
        # We compute how long ago (in frames) this shock occurred and decay
        shock_frame_pos = t_pos / len(window_dates) * total_frames
        frames_since = frame_idx - shock_frame_pos
        decay = np.exp(-frames_since / 300) if frames_since >= 0 else 0.0
        beam_alpha = float(np.clip(decay, 0.05, 0.9))

        # Draw vertical beam
        beam_z = np.linspace(z_floor, z_floor + z_range * 0.85, 10)
        beam_x = np.full_like(beam_z, t_pos)
        beam_y = np.full_like(beam_z, a_pos)

        ax.plot(beam_x, beam_y, beam_z,
                color=color, linewidth=1.5, alpha=beam_alpha, zorder=7)

        # Tip glow sphere
        ax.scatter(
            [t_pos], [a_pos], [z_floor + z_range * 0.85],
            s=40, color=color, alpha=beam_alpha * 0.8,
            zorder=8, depthshade=False,
        )


def _style_3d_axes(ax, window_dates, n_labels):
    """Apply axis labels, ticks, and limits to the 3D surface axes."""
    w = len(window_dates)
    a = n_labels

    ax.set_xlim(0, max(w - 1, 1))
    ax.set_ylim(0, max(a - 1, 1))
    ax.set_zlim(0, 1)

    # X ticks: every ~15 days, show abbreviated date
    x_tick_step = max(1, w // 6)
    x_ticks = list(range(0, w, x_tick_step))
    x_labels = []
    for i in x_ticks:
        if i < len(window_dates):
            d = window_dates[i]
            # Cross-platform: avoid %-m (Linux-only)
            x_labels.append(f"{d.month}/{str(d.year)[2:]}")
        else:
            x_labels.append("")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=7, color="#6688aa")

    # Y ticks: asset labels
    ax.set_yticks(list(range(a)))
    ax.set_yticklabels(ASSET_LABELS[:a], fontsize=8, color="#8899cc")

    # Z ticks
    ax.set_zticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_zticklabels(["0", ".25", ".5", ".75", "1"], fontsize=7, color="#6688aa")

    ax.set_xlabel("Time", color="#6688aa", fontsize=8, labelpad=5)
    ax.set_ylabel("Asset", color="#6688aa", fontsize=8, labelpad=5)
    ax.set_zlabel("Energy", color="#00e5ff", fontsize=8, labelpad=5)
