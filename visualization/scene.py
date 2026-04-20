"""
visualization/scene.py
───────────────────────
Create the Matplotlib figure and axes layout for the 2K portrait video frame.

Canvas: 12 × 16 inches @ 150 DPI  →  1800 × 2400 px rendered
        ↓ FFmpeg downscales to 1440 × 1920 (2K 3:4 portrait) during encode.

Portrait layout (top → bottom):
  ┌─────────────────────────────┐  (100%)
  │  HEADER HUD                 │   7%   ax_header
  ├─────────────────────────────┤
  │                             │
  │   3D ENERGY SURFACE (main)  │  57%   ax_3d
  │                             │
  ├─────────────────────────────┤
  │  ALPHA HEATMAP STRIP        │   6%   ax_heatmap
  ├─────────────────────────────┤
  │  PRICE + ENERGY TIME-SERIES │  22%   ax_ts   (twin on ax_ts2)
  ├─────────────────────────────┤
  │  FOOTER / LEGEND            │   8%   ax_footer
  └─────────────────────────────┘
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless — must be set before pyplot import

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path


# ── Design constants ────────────────────────────────────────────────────────
BG_COLOR    = "#050510"
ACCENT      = "#00e5ff"
DIM_GRID    = "#111133"
TEXT_COLOR  = "#e0e0ff"

# Render resolution
RENDER_W_IN   = 4.5     # figure width  (inches)
RENDER_H_IN   = 8.0     # figure height (inches)
RENDER_DPI    = 160     # → 720 × 1280 px  (720p portrait)

# Output resolution (passed to FFmpeg vf=scale)
OUT_W, OUT_H  = 720, 1280

# Frame rate
FPS = 120

# ── Font setup ──────────────────────────────────────────────────────────────

def _register_fonts() -> str:
    """
    Try to register Rajdhani Bold from the assets/ folder.
    Falls back to 'DejaVu Sans' if not found.
    """
    candidate = Path(__file__).parent.parent / "assets" / "Rajdhani-Bold.ttf"
    if candidate.exists():
        fm.fontManager.addfont(str(candidate))
        try:
            prop = fm.FontProperties(fname=str(candidate))
            name = prop.get_name()
            return name
        except Exception:
            pass
    return "DejaVu Sans"


FONT_FAMILY = _register_fonts()


# ── Figure factory ──────────────────────────────────────────────────────────

def create_figure() -> tuple:
    """
    Create and return the figure + all axes.

    Returns
    -------
    fig, ax_header, ax_3d, ax_heatmap, ax_ts, ax_footer
    """
    fig = plt.figure(
        figsize=(RENDER_W_IN, RENDER_H_IN),
        dpi=RENDER_DPI,
    )
    fig.patch.set_alpha(0.0)  # Make figure perfectly transparent for PyVista composite

    # GridSpec: 5 rows, proportional heights
    gs = fig.add_gridspec(
        5, 1,
        height_ratios=[7, 57, 6, 22, 8],
        hspace=0.0,
        left=0.05, right=0.97,
        top=0.98, bottom=0.02,
    )

    # ── Header ──────────────────────────────────────────────────────────
    ax_header = fig.add_subplot(gs[0])
    _style_flat_ax(ax_header)

    # ── 3D surface (now an invisible spacer for PyVista layer) ───────────
    ax_3d = fig.add_subplot(gs[1])
    ax_3d.set_facecolor((0, 0, 0, 0))
    ax_3d.axis('off')


    # ── Alpha heatmap strip ──────────────────────────────────────────────
    ax_heatmap = fig.add_subplot(gs[2])
    _style_flat_ax(ax_heatmap)

    # ── Time-series panel ────────────────────────────────────────────────
    ax_ts = fig.add_subplot(gs[3])
    _style_ts_ax(ax_ts)

    # ── Footer ───────────────────────────────────────────────────────────
    ax_footer = fig.add_subplot(gs[4])
    _style_flat_ax(ax_footer)

    return fig, ax_header, ax_3d, ax_heatmap, ax_ts, ax_footer


# ── Axis styling helpers ─────────────────────────────────────────────────────

def _style_flat_ax(ax):
    """Style a flat (2-D) axis: dark bg, no ticks."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _style_3d_ax(ax):
    """Apply neon dark theme to the 3D axes."""
    ax.set_facecolor(BG_COLOR)
    # Pane background: very dark navy with slight transparency
    pane_color = (0.02, 0.02, 0.08, 0.85)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    # Grid lines: dim
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = DIM_GRID
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis.label.set_color(TEXT_COLOR)
        axis.label.set_fontsize(9)
        for tick in axis.get_ticklabels():
            tick.set_color("#6688aa")
            tick.set_fontsize(7)
    ax.tick_params(colors="#6688aa", labelsize=7)


def _style_ts_ax(ax):
    """Style the time-series panel."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors="#6688aa", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(DIM_GRID)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
