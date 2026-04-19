"""
visualization/camera.py
────────────────────────
Cinematic camera schedule for the 3D axes.

Motion:  slow continuous azimuth rotation with gentle elevation oscillation.
No jarring cuts — the camera glides the full 60-second video.

  elevation(t) = 28 + 6 · sin(2π · t / T)           graceful dip-and-rise
  azimuth(t)   = 200 + 150 · (t / T)                 full slow orbit ~¾ turn
"""

from __future__ import annotations

import math
import numpy as np


def get_view(frame_idx: int, total_frames: int) -> tuple[float, float]:
    """
    Return the (elev, azim) camera angles for a given frame.

    Parameters
    ----------
    frame_idx    : current frame (0-based)
    total_frames : total number of frames in the video

    Returns
    -------
    elev : elevation angle in degrees  (positive = looking down)
    azim : azimuth angle in degrees    (rotation around Z)
    """
    t = frame_idx / max(total_frames - 1, 1)   # normalised 0 → 1

    # Elevation: oscillates between 22° and 34°
    elev = 28.0 + 6.0 * math.sin(2.0 * math.pi * t)

    # Azimuth: slow continuous orbit, 200° → 350° (150° sweep over 60s)
    azim = 200.0 + 150.0 * t

    return elev, azim


def get_intro_view(frame_idx: int, intro_frames: int = 120) -> tuple[float, float]:
    """
    During the first `intro_frames`, fly in from a high, close angle.
    Used to create a cinematic intro zoom.

    Returns (elev, azim) — call this if frame_idx < intro_frames,
    then blend with get_view() using smooth_step.
    """
    t = frame_idx / max(intro_frames - 1, 1)
    # Start: high overhead looking straight down; end: normal viewing angle
    elev = 70.0 * (1 - t) + 28.0 * t
    azim = 270.0 * (1 - t) + 200.0 * t
    return elev, azim


def get_blended_view(frame_idx: int, total_frames: int, intro_frames: int = 180) -> tuple[float, float]:
    """
    Blended camera: cinematic intro for first `intro_frames`, then normal orbit.

    Parameters
    ----------
    frame_idx    : current frame index
    total_frames : total frames in video
    intro_frames : frames spent in intro (default: 180 = 1.5 s at 120 fps)
    """
    if frame_idx < intro_frames:
        t_blend = frame_idx / intro_frames
        # Smooth step easing
        t_blend = t_blend * t_blend * (3 - 2 * t_blend)
        e_intro, a_intro = get_intro_view(frame_idx, intro_frames)
        # Target for blend end is the orbit start position
        _, a_orbit_start = get_view(0, total_frames)
        elev = e_intro * (1 - t_blend) + 28.0 * t_blend
        azim = a_intro * (1 - t_blend) + a_orbit_start * t_blend
    else:
        # Adjust frame index to account for intro
        adjusted = frame_idx - intro_frames
        adjusted_total = total_frames - intro_frames
        elev, azim = get_view(adjusted, adjusted_total)

    return elev, azim
