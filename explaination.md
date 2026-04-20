# Quantum Market Pulse — 4D Alpha Engine Animation Concept & Theory

This document outlines the core concepts, mathematical theories, and programmatic workflow behind the animated visualization created by the **Quantum Market Pulse — 4D Alpha Engine**. It is designed as a study note to understand how raw financial market data is transformed into a rich, physical 3D animation.

## 1. The Core Concept
The engine abandons traditional 2D candlestick charts in favor of a **4D Market Energy Field**. The animation treats the financial market as a fluid, dynamic landscape. Instead of simply graphing price over time, the engine generates an undulating, volumetric surface where:
*   The **X-axis** represents **Time**.
*   The **Y-axis** represents different **Assets**.
*   The **Z-axis (Height)** represents the absolute **"Market Energy"** at a given time and asset.
*   The **4th Dimension (Color / Intensity)** represents the **"Alpha"**—the directional momentum of the asset (Bullish/Bearish).

## 2. Theoretical Details & Mathematics

### The Market Energy Surface $E(t,a)$
The height of the landscape is driven by **Market Energy**, an aggregated metric that determines the "activity" or intensity of an asset.
It's calculated as a weighted scalar sum of four key normalized metrics:
*   **Volatility** (35% weight)
*   **Absolute Momentum** (25% weight)
*   **Volume Spikes** (25% weight)
*   **Sentiment** (15% weight)

$$E_{\text{raw}} = 0.35 \times \text{Vol} + 0.25 \times |\text{Mom}| + 0.25 \times \text{VolumeSpike} + 0.15 \times \text{Sentiment}$$

### Wavy Fluid Dynamics (Gaussian Smoothing)
To achieve the cinematic, liquid-like "TikTok" visual style, the raw energy field is processed through a **2D Gaussian Filter**. This smoothing algorithm operates across both the time dimension $\sigma_{\text{time}}$ and the asset dimension $\sigma_{\text{asset}}$, eliminating harsh jagged edges and making the market data roll smoothly like ocean waves.

### The Alpha Signal (Color Mapping)
A secondary metric, the Alpha array $\alpha(t, a)$, determines the skin of the 3D surface:
*   Calculated using a bounded hyperbolic tangent: 
    $$\alpha = \text{sign}(\text{momentum}) \times \tanh(4 \times |\text{normalized momentum}|)$$
*   This maps the directional strength exclusively between $[-1, 1]$.
*   **Neon Green ($>0$)**: Signifies bullish momentum (upward pressure).
*   **Neon Magenta ($<0$)**: Signifies bearish momentum (downward pressure).

### Anomaly / Shock Detection
The model flags statistical anomalies (Z-score $E > 2.0\sigma$) and extreme price shocks (log returns $> 3.0\sigma$). These logical points are layered onto the 3D grid as scattered markers with dual-rendered opacities—creating physical "glowing core" and "halo" effects on the topological surface.

## 3. How It Works: The Rendering Pipeline

The system uses a highly optimized, dual-engine rendering loop. To output smooth 120 FPS high-resolution (2K Portrait) video, it performs the following steps:

1. **GPU Topographical Rendering (PyVista)**: 
   * A headless OpenGL pipeline (via PyVista) generates a `StructuredGrid` from the $X$, $Y$, and $Z$ planes. 
   * The matrix is exponentially zoomed (`scipy.ndimage.zoom`) to artificially scale the Y-axis so the few assets occupy the same physical dimensions as the days (X-axis), locking it into a squared, dramatic presentation.
   * A camera is mathematically placed at a constrained $elev=35^\circ$, $azim=225^\circ$, pointing slightly below the mesh base to force a cinematic "upward" tilt layout ideal for mobile phone ratios.
2. **CPU UI Rendering (Matplotlib)**:
   * The 2D informational HUD—consisting of heatmaps, tracking labels, headers, and footers—is drawn to an invisible canvas on the CPU.
3. **Alpha-Compositing**:
   * The transparent 2D UI image overlay is mathematically alpha-blended bit-by-bit on top of the 3D frame returned from the GPU.
4. **Parallel Multiprocessing & FFmpeg**:
   * Isolated multiprocessing workers simultaneously crunch different frame batches. 
   * As frames are blended into `.rgb24` arrays, they are streamed sequentially into an OS-level spawned **FFmpeg** instance via standard input pipe `stdin`. FFmpeg automatically compresses this stream on the fly into an H.264 MP4 video.
