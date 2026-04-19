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
        "-c:v", "h264_nvenc",
        "-preset", "p6",
        "-cq", "18",
        "-rc", "vbr",
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

# ── Multiprocessing Worker ────────────────────────────────────────────────────
import concurrent.futures
import multiprocessing as mp

g_E = None
g_alpha_arr = None
g_anomaly = None
g_dates = None
g_shocks = None
g_price_df = None
g_weights = None
g_asset_names = None
g_window = None
g_total_frames = None
g_T = None

def _init_worker(E, alpha_arr, anomaly, dates, shocks, price_df, weights, asset_names, window, total_frames, T):
    global g_E, g_alpha_arr, g_anomaly, g_dates, g_shocks, g_price_df, g_weights, g_asset_names, g_window, g_total_frames, g_T
    global pv_engine  # PyVista GPU Engine instance
    g_E = E
    g_alpha_arr = alpha_arr
    g_anomaly = anomaly
    g_dates = dates
    g_shocks = shocks
    g_price_df = price_df
    g_weights = weights
    g_asset_names = asset_names
    g_window = window
    g_total_frames = total_frames
    g_T = T

    # Initialize the PyVista headless NVIDIA GPU Engine per child process
    from visualization.pyvista_renderer import PyVistaGPU
    # Set to matched high-res 1800x2400 dimensions for pixel-perfect layout alignment
    from visualization.scene import RENDER_W_IN, RENDER_H_IN, RENDER_DPI
    w_px = int(RENDER_W_IN * RENDER_DPI)
    h_px = int(RENDER_H_IN * RENDER_DPI)
    pv_engine = PyVistaGPU(w_px=w_px, h_px=h_px)


def _render_frame_worker(frame_idx: int) -> tuple[int, bytes]:
    """Generates a single frame in an isolated process. Returns (frame_idx, rgb24_bytes)."""
    import matplotlib.pyplot as plt
    from visualization.scene import create_figure, _style_3d_ax
    from visualization.camera import get_blended_view
    from visualization.overlays import draw_all_overlays
    
    t_frac = frame_idx / max(g_total_frames - 1, 1)
    t_idx  = int(g_window + t_frac * (g_T - 1 - g_window))
    t_idx  = min(t_idx, g_T - 1)

    # 1. RENDER 3D FRAME ON NVIDIA GPU (PyVista)
    t_start = max(0, t_idx - g_window)
    t_end   = t_idx + 1
    
    E_win     = g_E[t_start:t_end, :]
    alpha_win = g_alpha_arr[t_start:t_end, :]
    anom_win  = g_anomaly[t_start:t_end, :]
    
    img_3d_gpu = pv_engine.render_3d_frame(E_win, alpha_win, anom_win)

    # 2. RENDER 2D UI ON CPU (Matplotlib)
    fig, ax_header, _, ax_heatmap, ax_ts, ax_footer = create_figure()

    draw_all_overlays(
        ax_header=ax_header,
        ax_heatmap=ax_heatmap,
        ax_ts=ax_ts,
        ax_footer=ax_footer,
        price_df=g_price_df,
        E=g_E,
        alpha_arr=g_alpha_arr,
        dates=g_dates,
        t_idx=t_idx,
        frame_idx=frame_idx,
        total_frames=g_total_frames,
        weights=g_weights,
        asset_names=g_asset_names,
    )

    fig.canvas.draw()
    img_2d_ui = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    
    from visualization.scene import RENDER_W_IN, RENDER_H_IN, RENDER_DPI
    w_px = int(RENDER_W_IN * RENDER_DPI)
    h_px = int(RENDER_H_IN * RENDER_DPI)
    img_2d_ui = img_2d_ui.reshape(h_px, w_px, 4)

    plt.close(fig)
    import gc
    gc.collect()

    # 3. ALPHA-COMPOSITE BLEND
    # img_2d_ui has transparent background, so it flawlessly overlays img_3d_gpu
    alpha_fg = img_2d_ui[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg

    final_img = np.empty((h_px, w_px, 3), dtype=np.uint8)
    for c in range(3):
        final_img[:, :, c] = (alpha_fg * img_2d_ui[:, :, c] + alpha_bg * img_3d_gpu[:, :, c]).astype(np.uint8)

    buf = np.ascontiguousarray(final_img).tobytes()

    return frame_idx, buf


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

    # ── FFmpeg pipe ────────────────────────────────────────────────────────
    ffmpeg_proc = open_ffmpeg_pipe(output_path, args.fps)

    t0 = time.time()
    rendered = 0

    max_workers = os.cpu_count() or 4
    frames_list = list(frames_to_render)
    next_expected_i = 0
    frame_buffer = {}

    print(f"🚀  Using {max_workers} processes for parallel rendering.\n")

    try:
        with mp.Pool(
            processes=max_workers,
            initializer=_init_worker,
            initargs=(E, alpha_arr, anomaly, dates, shocks, price_df, weights, asset_names, args.window, total_frames, T),
            maxtasksperchild=10
        ) as pool:
            
            for f_idx, buf in pool.imap_unordered(_render_frame_worker, frames_list):
                frame_buffer[f_idx] = buf
                
                # Write sequentially
                while next_expected_i < len(frames_list) and frames_list[next_expected_i] in frame_buffer:
                    expected_f_idx = frames_list[next_expected_i]
                    ffmpeg_proc.stdin.write(frame_buffer.pop(expected_f_idx))
                    
                    rendered += 1
                    if rendered % 30 == 0 or rendered == 1:
                        elapsed = time.time() - t0
                        fps_actual = rendered / max(elapsed, 1e-3)
                        remaining  = (n_out - rendered) / max(fps_actual, 0.001)
                        pct = 100 * rendered / n_out
                        print(
                            f"\r  [{pct:5.1f}%]  frame {expected_f_idx:5d}  |  "
                            f"{fps_actual:.2f} fr/s  |  ETA {remaining/60:.1f} min",
                            end="", flush=True,
                        )
                    next_expected_i += 1

    except BrokenPipeError:
        print("\n⚠️  FFmpeg pipe closed unexpectedly. File may be truncated.")
    finally:
        try:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait(timeout=60)
        except Exception:
            ffmpeg_proc.kill()

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
