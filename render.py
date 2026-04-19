"""
render.py — Quantum Market Pulse · 4D Alpha Engine
════════════════════════════════════════════════════
Main entry point. Produces a cinema-quality MP4 video of the 3D
market energy field evolving over time.

Output
------
  output/quantum_market_pulse.mp4
    Resolution : 1440 × 1920 px  (2K portrait, 3:4)
    Frame rate : 120 fps
    Duration   : ~60 seconds (7 200 frames)
    Codec      : H.264, CRF 18, yuv420p

Usage
-----
  python render.py                  # full 60-second render
  python render.py --preview        # quick 240-frame preview (~2 s @ 120fps)
  python render.py --duration 30    # 30-second version
  python render.py --output my.mp4  # custom output path
  python render.py --no-yfinance    # force offline / synthetic assets

Requirements
------------
  pip install -r requirements.txt
  FFmpeg must be on PATH  →  winget install Gyan.FFmpeg
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Project-root sys.path fix (run from any working directory) ───────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── Local imports ─────────────────────────────────────────────────────────────
from data.fetch_data import load_data
from features.compute_features import compute_features
from models.energy_model import compute_energy, identify_shocks, DEFAULT_WEIGHTS
from visualization.scene import (
    create_figure, RENDER_W_IN, RENDER_H_IN, RENDER_DPI, OUT_W, OUT_H, FPS
)
from visualization.camera import get_blended_view
from visualization.surface_renderer import draw_surface
from visualization.overlays import draw_all_overlays


# ── Constants ────────────────────────────────────────────────────────────────
OUTPUT_DIR   = PROJECT_ROOT / "output"
RENDER_W_PX  = int(RENDER_W_IN * RENDER_DPI)   # 1800
RENDER_H_PX  = int(RENDER_H_IN * RENDER_DPI)   # 2400


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantum Market Pulse — 4D Alpha Engine video renderer"
    )
    p.add_argument("--preview", action="store_true",
                   help="Render every 30th frame only (~240 frames → output/preview.mp4)")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Video duration in seconds (default: 60)")
    p.add_argument("--fps", type=int, default=FPS,
                   help=f"Frame rate (default: {FPS})")
    p.add_argument("--output", type=str, default=None,
                   help="Output file path (default: output/quantum_market_pulse.mp4)")
    p.add_argument("--start-date", type=str, default="2017-01-01",
                   help="Data start date (default: 2017-01-01)")
    p.add_argument("--no-yfinance", action="store_true",
                   help="Skip yfinance fetch; use synthetic assets only")
    p.add_argument("--window", type=int, default=90,
                   help="Days shown on 3D surface at one time (default: 90)")
    return p.parse_args()


# ── FFmpeg subprocess ─────────────────────────────────────────────────────────

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("\n❌  ffmpeg not found on PATH.")
        print("    Install via:  winget install Gyan.FFmpeg")
        print("    Then restart your terminal and try again.\n")
        sys.exit(1)


def open_ffmpeg_pipe(output_path: Path, fps: int) -> subprocess.Popen:
    """Open an FFmpeg process consuming raw RGB24 frames from stdin."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{RENDER_W_PX}x{RENDER_H_PX}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:",
        "-vf", f"scale={OUT_W}:{OUT_H}",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


# ── Main render loop ──────────────────────────────────────────────────────────

def render(args: argparse.Namespace) -> None:
    print("\n" + "═" * 58)
    print("  QUANTUM MARKET PULSE — 4D Alpha Engine")
    print("  Video Renderer  ·  2K Portrait 3:4  ·  120 fps")
    print("═" * 58 + "\n")

    check_ffmpeg()
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_frames  = int(args.duration * args.fps)
    preview_step  = 30 if args.preview else 1
    frames_to_render = range(0, total_frames, preview_step)

    if args.preview:
        n_out = len(list(frames_to_render))
        out_name = "preview.mp4"
        print(f"📽️   PREVIEW mode: {n_out} frames → {args.fps // preview_step} fps equivalent")
    else:
        n_out = total_frames
        out_name = "quantum_market_pulse.mp4"
        print(f"📽️   FULL render: {total_frames} frames @ {args.fps} fps ({args.duration:.0f}s)")

    output_path = Path(args.output) if args.output else OUTPUT_DIR / out_name
    print(f"💾  Output: {output_path}\n")

    # ── Data pipeline ─────────────────────────────────────────────────────
    if args.no_yfinance:
        os.environ["NO_YFINANCE"] = "1"

    price_df, volume_df = load_data(start_date=args.start_date)
    asset_names         = list(price_df.columns)

    features = compute_features(price_df, volume_df)

    weights = DEFAULT_WEIGHTS.copy()
    E, alpha_arr, anomaly = compute_energy(features, weights=weights)

    shocks = identify_shocks(price_df)

    dates = price_df.index
    T     = len(dates)

    print(f"\n🎬  Starting render → {output_path.name}")
    print(f"    Frame size: {RENDER_W_PX} × {RENDER_H_PX} px (render)  →  {OUT_W} × {OUT_H} px (output)")
    print(f"    Frames: {n_out}  |  Window: {args.window} days\n")

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, ax_header, ax_3d, ax_heatmap, ax_ts, ax_footer = create_figure()

    # ── FFmpeg pipe ────────────────────────────────────────────────────────
    ffmpeg_proc = open_ffmpeg_pipe(output_path, args.fps)

    t0 = time.time()
    rendered = 0

    try:
        for frame_idx in frames_to_render:
            # Map frame index → data index (0 → window_size, total_frames-1 → T-1)
            t_frac = frame_idx / max(total_frames - 1, 1)
            t_idx  = int(args.window + t_frac * (T - 1 - args.window))
            t_idx  = min(t_idx, T - 1)

            # ── Camera ────────────────────────────────────────────────────
            elev, azim = get_blended_view(frame_idx, total_frames, intro_frames=180)

            # ── 3D surface (clear + redraw) ───────────────────────────────
            ax_3d.cla()
            from visualization.scene import _style_3d_ax
            _style_3d_ax(ax_3d)

            draw_surface(
                ax=ax_3d,
                E=E,
                alpha_arr=alpha_arr,
                anomaly=anomaly,
                dates=dates,
                t_idx=t_idx,
                shocks=shocks,
                frame_idx=frame_idx,
                total_frames=total_frames,
                window=args.window,
            )

            ax_3d.view_init(elev=elev, azim=azim)

            # ── 2D overlays (in-place update) ─────────────────────────────
            draw_all_overlays(
                ax_header=ax_header,
                ax_heatmap=ax_heatmap,
                ax_ts=ax_ts,
                ax_footer=ax_footer,
                price_df=price_df,
                E=E,
                alpha_arr=alpha_arr,
                dates=dates,
                t_idx=t_idx,
                frame_idx=frame_idx,
                total_frames=total_frames,
                weights=weights,
                asset_names=asset_names,
            )

            # ── Render to raw RGB bytes → FFmpeg ──────────────────────────
            fig.canvas.draw()
            # matplotlib 3.8+: tostring_rgb removed; use buffer_rgba and strip alpha
            rgba_buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            rgba_buf = rgba_buf.reshape(RENDER_H_PX, RENDER_W_PX, 4)
            buf = np.ascontiguousarray(rgba_buf[:, :, :3])  # drop alpha → RGB24
            ffmpeg_proc.stdin.write(buf.tobytes())

            rendered += 1
            if rendered % 30 == 0 or rendered == 1:
                elapsed = time.time() - t0
                fps_actual = rendered / max(elapsed, 1e-3)
                remaining  = (n_out - rendered) / max(fps_actual, 0.001)
                pct = 100 * rendered / n_out
                print(
                    f"\r  [{pct:5.1f}%]  frame {frame_idx:5d}  |  "
                    f"{fps_actual:.2f} fr/s  |  ETA {remaining/60:.1f} min",
                    end="", flush=True,
                )

    except BrokenPipeError:
        print("\n⚠️  FFmpeg pipe closed unexpectedly. File may be truncated.")
    finally:
        try:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait(timeout=60)
        except Exception:
            ffmpeg_proc.kill()
        plt.close(fig)

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1e6 if output_path.exists() else 0

    print(f"\n\n✅  Saved: {output_path}")
    print(f"   Frames: {rendered}  |  Time: {elapsed/60:.1f} min  |  Size: {size_mb:.1f} MB")
    print(f"   Resolution: {OUT_W}×{OUT_H}  |  FPS: {args.fps}")
    if args.preview:
        print("\n   Run without --preview for the full 60-second render.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    render(args)
