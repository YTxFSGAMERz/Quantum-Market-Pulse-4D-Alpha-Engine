# Quantum Market Pulse — 4D Alpha Engine

> **High-Performance TikTok-Style Quantitative Animation Engine** — The global market rendered as a living, 3D energy topology via PyVista GPU acceleration.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YTxFSGAMERz/Quantum-Market-Pulse-4D-Alpha-Engine/blob/main/Colab.ipynb)

## Output Specifications

| | |
|---|---|
| Engine | **PyVista (GPU) + Matplotlib (CPU) Hybrid Compositor** |
| Resolution | **720 × 1280** (720p portrait, 9:16 — TikTok/Instagram native) |
| Frame rate | **120 fps** |
| Duration | **~60 seconds** |
| Data | ~3400 trading days (BTC, ETH, SPY, GOLD) |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

*(Note: PyVista utilizes hardware GPU acceleration. On headless Linux/Colab, `xvfb` is required and will be started automatically via `pv.start_xvfb()`.)*

### 2. Install FFmpeg (required to compile the video)

**Windows:**
```bash
winget install Gyan.FFmpeg
```

**Linux / Colab:**
```bash
sudo apt-get install ffmpeg xvfb
```

### 3. Preview render

Generates a 2-second clip equivalent (at 120fps) to validate the layout and topological geometry.
```bash
python render.py --preview
```

### 4. Full GPU Render (~15-20 minutes)

```bash
python render.py
# → output/quantum_market_pulse.mp4
```

## CLI Options

```
python render.py [OPTIONS]

  --preview         Render a short fraction of frames for quick visual validation
  --duration SECS   Video length in seconds  (default: 60)
  --fps N           Frame rate  (default: 120)
  --output PATH     Custom output file path
  --start-date DATE Data start date  (default: 2017-01-01)
  --no-yfinance     Force offline mode, use synthetic sibling assets only
  --window DAYS     Days shown on the moving 3D topological surface (default: 90)
```

## Advanced Hybrid Architecture

The engine previously relied purely on Matplotlib CPU 3D projections, taking upwards of 10 hours. It has been entirely re-engineered for parallel memory management and hardware speeds:

```
project_root/
├── data/
│   └── fetch_data.py            # Local BTC CSV + dynamic yfinance (ETH/SPY/GOLD) with synthesis fallback
├── features/
│   └── compute_features.py      # Computes vol, momentum, RSI, sentiment into massive arrays
├── models/
│   └── energy_model.py          # E(t,a) 4D surface energy matrix · directional α signal
├── visualization/
│   ├── pyvista_renderer.py      # [GPU] Renders 90x90 interpolated topological 3D meshes and axes via VTK
│   ├── overlays.py              # [CPU] Draws 720p portrait HUD elements (heatmaps, timeseries)
│   ├── camera.py                # Mathematical orbits and focal panning
│   └── scene.py                 # Core canvas matrix, DPI, and layout scaling geometry
├── render.py                    # Multi-core frame compositor linking GPU/CPU layers into FFmpeg
└── output/                      # Target render directory
```

## Topological 3D Mechanics

* **Fluid Interpolation:** The raw 90×4 data matrix is upscaled dynamically via `scipy.ndimage.zoom()` into a dense 90×90 wave topological surface, providing an aerodynamic, fluid aesthetic.
* **Dimensional Boundaries:** The PyVista environment injects rigid floating mathematical boundary frames, fixing the visual axes precisely down to custom metric scales.
* **Energy Scale:** 

```
E(t, a) = 0.35 · volatility + 0.25 · |momentum| + 0.25 · volume_spike + 0.15 · sentiment
```

* **Color Physics (`α(t,a) = sign(momentum) × tanh(4 × |momentum|)`):**
  - **Green** (α > 0) → Bullish acceleration
  - **Magenta** (α < 0) → Bearish velocity
  - **Deep blue** (α ≈ 0) → Stagnation point

## Fonts (Optional for Windows)

To achieve the highly engineered numeric aesthetic, the engine prioritizes `Rajdhani-Bold.ttf`.
1. Download from Google Fonts: https://fonts.google.com/specimen/Rajdhani
2. Place `Rajdhani-Bold.ttf` in an `assets/` folder in the root path.
*(Falls back to DejaVu Sans natively if undetected).*
