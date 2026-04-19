"""
visualization/overlays.py
──────────────────────────
Draw all 2D HUD, time-series, and heatmap elements onto the figure.

These are updated in-place each frame without clearing the figure —
only the individual axes are cleared and redrawn, keeping the 3D
axes independent.

Components:
  • ax_header  — Title, subtitle, live date, energy readout per asset
  • ax_heatmap — α(t,a) rolling 60-day strip heatmap
  • ax_ts      — BTC price line + market energy area fill + scan-line cursor
  • ax_footer  — legend, weights display, frame counter
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from visualization.scene import BG_COLOR, ACCENT, TEXT_COLOR, DIM_GRID, FONT_FAMILY
from utils.helpers import NEON_ALPHA_CMAP


# ── Colour palette ────────────────────────────────────────────────────────────
BULL_COLOR = "#39ff14"
BEAR_COLOR = "#ff0066"
NEUT_COLOR = "#aa44ff"
PRICE_COLOR = "#ffffff"
ENERGY_COLOR = "#00e5ff"
ASSET_COLORS = ["#00e5ff", "#aa44ff", "#ffd700", "#ff6600"]


# ── Header ────────────────────────────────────────────────────────────────────

def draw_header(ax_header, dates: pd.DatetimeIndex, t_idx: int,
                E: np.ndarray, alpha_arr: np.ndarray,
                asset_names: list[str]) -> None:
    """Render the top HUD: title + live asset energy ticker."""
    ax_header.cla()
    ax_header.set_facecolor(BG_COLOR)
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    ax_header.axis("off")

    current_date = dates[min(t_idx, len(dates) - 1)]

    # ── Title ──────────────────────────────────────────────────────────────
    ax_header.text(
        0.03, 0.82, "QUANTUM MARKET PULSE",
        fontsize=18, fontweight="bold", fontfamily=FONT_FAMILY,
        color=TEXT_COLOR, va="top", ha="left", transform=ax_header.transAxes,
    )
    ax_header.text(
        0.03, 0.50, "4D ALPHA ENGINE  ·  ENERGY FIELD ANALYSIS",
        fontsize=9, fontfamily=FONT_FAMILY,
        color=ACCENT, va="top", ha="left", transform=ax_header.transAxes,
        alpha=0.85,
    )

    # ── Date readout ───────────────────────────────────────────────────────
    date_str = current_date.strftime("%d %b %Y")
    ax_header.text(
        0.97, 0.82, date_str,
        fontsize=14, fontweight="bold", fontfamily=FONT_FAMILY,
        color=ACCENT, va="top", ha="right", transform=ax_header.transAxes,
    )

    # ── Per-asset energy bullets ───────────────────────────────────────────
    e_vals  = E[min(t_idx, E.shape[0] - 1), :]
    a_vals  = alpha_arr[min(t_idx, alpha_arr.shape[0] - 1), :]
    n       = min(len(asset_names), len(e_vals))
    x_start = 0.60

    for i in range(n):
        col_dir = BULL_COLOR if a_vals[i] > 0.1 else (BEAR_COLOR if a_vals[i] < -0.1 else NEUT_COLOR)
        arrow   = "▲" if a_vals[i] > 0.05 else ("▼" if a_vals[i] < -0.05 else "◆")
        xpos    = x_start + i * 0.095

        ax_header.text(
            xpos, 0.45, asset_names[i],
            fontsize=7, color="#8899cc", fontfamily=FONT_FAMILY,
            va="center", ha="center", transform=ax_header.transAxes,
        )
        ax_header.text(
            xpos, 0.18, f"{e_vals[i]:.2f}",
            fontsize=9, fontweight="bold", color=col_dir,
            fontfamily=FONT_FAMILY, va="center", ha="center", transform=ax_header.transAxes,
        )
        ax_header.text(
            xpos + 0.025, 0.18, arrow,
            fontsize=7, color=col_dir,
            va="center", ha="left", transform=ax_header.transAxes,
        )

    # Thin accent divider below header
    ax_header.axhline(0.0, color=ACCENT, linewidth=0.6, alpha=0.5)


# ── Alpha Heatmap Strip ───────────────────────────────────────────────────────

def draw_heatmap(ax_heatmap, alpha_arr: np.ndarray, t_idx: int,
                 asset_names: list[str], strip_days: int = 60) -> None:
    """
    Render a rolling α(t,a) heatmap strip:
    X = last `strip_days` days  |  Y = assets (one row each)
    Color = RdYlGn mapped from [-1, 1]
    """
    ax_heatmap.cla()
    ax_heatmap.set_facecolor(BG_COLOR)

    t_end   = min(t_idx + 1, alpha_arr.shape[0])
    t_start = max(0, t_end - strip_days)
    strip   = alpha_arr[t_start:t_end, :].T   # (A, W)

    if strip.shape[1] < 2:
        return

    ax_heatmap.imshow(
        strip,
        aspect="auto",
        origin="lower",
        cmap=NEON_ALPHA_CMAP,
        vmin=-1, vmax=1,
        interpolation="nearest",
    )

    n = min(len(asset_names), strip.shape[0])
    ax_heatmap.set_yticks(range(n))
    ax_heatmap.set_yticklabels(asset_names[:n], fontsize=7, color="#8899cc")
    ax_heatmap.set_xticks([])

    ax_heatmap.set_ylabel("α  HEATMAP", color=ACCENT, fontsize=7, labelpad=3)

    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)
    ax_heatmap.tick_params(left=False, bottom=False)


# ── Time-series Panel ─────────────────────────────────────────────────────────

def draw_timeseries(ax_ts, price_df: pd.DataFrame, E: np.ndarray,
                    dates: pd.DatetimeIndex, t_idx: int) -> None:
    """
    Dual-axis panel:
      — Left  axis: BTC price (white line)
      — Right axis: aggregate market energy (cyan area fill)
    A vertical neon scan-line marks the current frame position.
    """
    ax_ts.cla()
    ax_ts.set_facecolor(BG_COLOR)

    n = len(dates)
    xs = np.arange(n)
    cur = min(t_idx, n - 1)

    # ── BTC price ──────────────────────────────────────────────────────────
    btc = price_df["BTC"].values if "BTC" in price_df.columns else price_df.iloc[:, 0].values
    btc_norm = (btc - btc.min()) / (btc.max() - btc.min() + 1e-9)

    ax_ts.plot(xs[:cur+1], btc_norm[:cur+1], color=PRICE_COLOR, linewidth=0.8, alpha=0.9, zorder=3)
    ax_ts.set_ylabel("Price (norm)", color=TEXT_COLOR, fontsize=7)
    ax_ts.set_ylim(-0.05, 1.15)

    # ── Aggregate energy ───────────────────────────────────────────────────
    ax_e = ax_ts.twinx()
    energy_mean = E.mean(axis=1)   # (T,)

    ax_e.fill_between(xs[:cur+1], 0, energy_mean[:cur+1], alpha=0.35, color=ENERGY_COLOR, zorder=2)
    ax_e.plot(xs[:cur+1], energy_mean[:cur+1], color=ENERGY_COLOR, linewidth=0.7, alpha=0.85, zorder=3)
    ax_e.set_ylabel("Energy", color=ENERGY_COLOR, fontsize=7)
    ax_e.set_ylim(0, 1.3)
    ax_e.tick_params(colors=ENERGY_COLOR, labelsize=6)
    ax_e.spines["right"].set_color(ENERGY_COLOR)
    ax_e.spines["right"].set_alpha(0.4)
    for sp in ["top", "left", "bottom"]:
        ax_e.spines[sp].set_visible(False)

    # ── Current frame scan-line ────────────────────────────────────────────
    cur = min(t_idx, n - 1)
    ax_ts.axvline(cur, color=ACCENT, linewidth=1.2, alpha=0.9, zorder=5)

    # Highlight dot on price
    ax_ts.scatter([cur], [btc_norm[cur]], color=ACCENT, s=18, zorder=6, edgecolors="white", linewidths=0.5)

    # ── X-axis year labels ─────────────────────────────────────────────────
    year_ticks = []
    year_labels = []
    prev_year = None
    for i, d in enumerate(dates):
        if d.year != prev_year:
            year_ticks.append(i)
            year_labels.append(str(d.year))
            prev_year = d.year

    ax_ts.set_xticks(year_ticks)
    ax_ts.set_xticklabels(year_labels, fontsize=7, color="#6688aa")
    ax_ts.set_xlim(0, n - 1)

    # Style
    for sp in ["top", "right"]:
        ax_ts.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax_ts.spines[sp].set_color(DIM_GRID)
    ax_ts.tick_params(colors="#6688aa", labelsize=6)


# ── Footer ────────────────────────────────────────────────────────────────────

def draw_footer(ax_footer, weights: dict, frame_idx: int, total_frames: int,
                asset_names: list[str]) -> None:
    """Render the footer legend: weight config + asset legend + frame progress."""
    ax_footer.cla()
    ax_footer.set_facecolor(BG_COLOR)
    ax_footer.set_xlim(0, 1)
    ax_footer.set_ylim(0, 1)
    ax_footer.axis("off")

    # Thin accent divider above footer
    ax_footer.axhline(1.0, color=ACCENT, linewidth=0.6, alpha=0.4)

    # ── Weight display ──────────────────────────────────────────────────────
    w_text = "  ·  ".join([f"{k[:3].upper()}={v:.2f}" for k, v in weights.items()])
    ax_footer.text(
        0.03, 0.65, f"W  {w_text}",
        fontsize=7, color="#6688aa", fontfamily=FONT_FAMILY,
        va="center", ha="left", transform=ax_footer.transAxes,
    )

    # ── Asset legend dots ───────────────────────────────────────────────────
    n = min(len(asset_names), len(ASSET_COLORS))
    for i in range(n):
        xpos = 0.03 + i * 0.12
        ax_footer.plot(
            [xpos + 0.005], [0.25], "o",
            markersize=5, color=ASSET_COLORS[i],
            transform=ax_footer.transAxes,
        )
        ax_footer.text(
            xpos + 0.018, 0.25, asset_names[i],
            fontsize=7, color=ASSET_COLORS[i], fontfamily=FONT_FAMILY,
            va="center", ha="left", transform=ax_footer.transAxes,
        )

    # ── Frame progress bar ──────────────────────────────────────────────────
    progress = frame_idx / max(total_frames - 1, 1)
    bar_x0, bar_y, bar_w, bar_h = 0.55, 0.30, 0.42, 0.12
    # Background track
    ax_footer.add_patch(mpatches.FancyBboxPatch(
        (bar_x0, bar_y - bar_h / 2), bar_w, bar_h,
        boxstyle="round,pad=0.005", facecolor="#111133", edgecolor=DIM_GRID,
        linewidth=0.5, transform=ax_footer.transAxes, zorder=2,
    ))
    # Fill
    if progress > 0.01:
        ax_footer.add_patch(mpatches.FancyBboxPatch(
            (bar_x0, bar_y - bar_h / 2), bar_w * progress, bar_h,
            boxstyle="round,pad=0.005", facecolor=ACCENT, edgecolor="none",
            transform=ax_footer.transAxes, zorder=3,
        ))
    # Label
    ax_footer.text(
        0.98, 0.30, f"Frame {frame_idx:05d}/{total_frames}",
        fontsize=6, color="#6688aa", va="center", ha="right",
        transform=ax_footer.transAxes,
    )

    ax_footer.text(
        0.03, 0.88,
        "▪ BULLISH  ▪ BEARISH  ▪ NEUTRAL",
        fontsize=6, color="#555577", va="top", ha="left",
        transform=ax_footer.transAxes,
    )


# ── Master draw call ──────────────────────────────────────────────────────────

def draw_all_overlays(
    ax_header, ax_heatmap, ax_ts, ax_footer,
    price_df: pd.DataFrame,
    E: np.ndarray,
    alpha_arr: np.ndarray,
    dates: pd.DatetimeIndex,
    t_idx: int,
    frame_idx: int,
    total_frames: int,
    weights: dict,
    asset_names: list[str],
) -> None:
    """Convenience wrapper: draw all overlay panels in one call."""
    draw_header(ax_header, dates, t_idx, E, alpha_arr, asset_names)
    draw_heatmap(ax_heatmap, alpha_arr, t_idx, asset_names)
    draw_timeseries(ax_ts, price_df, E, dates, t_idx)
    draw_footer(ax_footer, weights, frame_idx, total_frames, asset_names)
