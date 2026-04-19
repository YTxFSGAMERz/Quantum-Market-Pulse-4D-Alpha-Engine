# Quantum Market Pulse — 4D Alpha Engine

> **Cinematic 2K quantitative finance video** — the global market rendered as a living 3D energy field.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YTxFSGAMERz/Quantum-Market-Pulse-4D-Alpha-Engine/blob/main/Colab.ipynb)

## Output

| | |
|---|---|
| Resolution | **1440 × 1920** (2K portrait, 3:4 — Instagram-native) |
| Frame rate | **120 fps** |
| Duration | **~60 seconds** |
| Format | H.264 MP4, CRF 18 |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg (required — not a Python package)

```bash
winget install Gyan.FFmpeg
# Then restart your terminal
```

### 3. Preview render (~5 minutes)

```bash
python render.py --preview
# → output/preview.mp4  (2-second clip at 120fps)
```

### 4. Full render (~4-10 hours on CPU)

```bash
python render.py
# → output/quantum_market_pulse.mp4
```

## CLI Options

```
python render.py [OPTIONS]

  --preview         Render every 30th frame only for quick validation
  --duration SECS   Video length in seconds  (default: 60)
  --fps N           Frame rate  (default: 120)
  --output PATH     Custom output file path
  --start-date DATE Data start date  (default: 2017-01-01)
  --no-yfinance     Force offline mode, use synthetic sibling assets only
  --window DAYS     Days shown on 3D surface  (default: 90)
```

## Architecture

```
project_root/
├── data/fetch_data.py          # BTC CSV + yfinance (ETH/SPY/GOLD) with offline fallback
├── features/compute_features.py # log_return, volatility, momentum, RSI, vol_spike, sentiment
├── models/energy_model.py      # E(t,a) energy surface · α directional signal · anomaly detection
├── visualization/
│   ├── scene.py                # 1800×2400 Matplotlib canvas · 5-panel portrait layout
│   ├── camera.py               # Cinematic intro fly-in + slow azimuth orbit
│   ├── surface_renderer.py     # 3D plot_surface · neon facecolors · glow markers · shock beams
│   └── overlays.py             # HUD header · α heatmap strip · price+energy timeseries · footer
├── render.py                   # Main entry point → FFmpeg pipe → MP4
├── utils/helpers.py
├── btc-usd-max.csv             # Bundled BTC historical data (2013–present)
└── output/                     # Generated video files
```

## Energy Model

```
E(t, a) = 0.35 · volatility
         + 0.25 · |momentum|
         + 0.25 · volume_spike
         + 0.15 · sentiment
```

Color:  `α(t,a) = sign(momentum) × tanh(4 × |momentum|) ∈ [−1, 1]`
- **Green** (α > 0) → bullish
- **Magenta** (α < 0) → bearish
- **Deep blue/violet** (α ≈ 0) → neutral

## Data Sources

- **BTC** — bundled `btc-usd-max.csv` (CoinGecko daily, 2013–present)
- **ETH, SPY, GOLD** — fetched via `yfinance` at startup; replaced with correlated synthetic series if offline

## Fonts (optional)

For the best look, place `Rajdhani-Bold.ttf` in the `assets/` folder:

```bash
mkdir assets
# Download from: https://fonts.google.com/specimen/Rajdhani
# Move Rajdhani-Bold.ttf → assets/Rajdhani-Bold.ttf
```

Falls back to DejaVu Sans automatically if not found.
