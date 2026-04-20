import numpy as np
import pyvista as pv
import scipy.ndimage
from visualization.surface_renderer import _build_facecolors
from utils.helpers import NEON_ALPHA_CMAP

import sys

if sys.platform == "linux" or sys.platform == "linux2":
    try:
        pv.start_xvfb()  # crucial for linux/colab headless OpenGL
    except Exception:
        pass

class PyVistaGPU:
    def __init__(self, w_px=1440, h_px=1920):
        # We start an off-screen plotter.
        # This will query the NVIDIA driver or Xvfb for rendering context
        self.plotter = pv.Plotter(off_screen=True, window_size=[w_px, h_px])
        
        self.plotter.set_background(color="black")
        self.plotter.image_transparent_background = True

        # Matches Matplotlib's exact elev=45, azim=225 setup
        # PyVista camera pos setup requires mathing bounding boxes or using angles
        # We will dynamically set the camera position when the first mesh is drawn!
        self.first_render = True

    def render_3d_frame(self, E_win, alpha_win, anom_win) -> np.ndarray:
        """
        Renders the highly mathematical 3D structures purely on the GPU.
        Returns a Numpy RGBA (1440, 1920, 4) uncompressed transparent image.
        """
        # We MUST clear previous meshes purely.
        self.plotter.clear()

        # Build Mesh Grid Data (Float32 to prevent PyVista warnings)
        W, A = E_win.shape
        
        # Interpolate drastically along the Asset (Y) dimension to make it wavy and smooth (TikTok style)
        import scipy.ndimage
        target_A = max(W, 90) # square up the grid
        zoom_y = target_A / max(A, 1)
        
        E_smooth = scipy.ndimage.zoom(E_win, (1.0, zoom_y), order=3)
        alpha_smooth = scipy.ndimage.zoom(alpha_win, (1.0, zoom_y), order=3)

        # Scale factors to make the visualization proportionally volumetric
        Z_SCALE = 50.0

        x = np.arange(W, dtype=np.float32)
        y = np.arange(target_A, dtype=np.float32) * (float(W) / target_A)  # Maps Y to the same physical size as X
        X, Y = np.meshgrid(x, y) 
        Z = E_smooth.T.astype(np.float32) * Z_SCALE

        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        
        # We must re-shape facecolors to point scalars
        # Matplotlib uses a per-vertex approach
        facecolors = _build_facecolors(E_smooth, alpha_smooth) # (target_A, W, 4)
        flat_colors = facecolors.reshape(-1, 4)
        
        # Bind the scalars
        grid.point_data["colors"] = (flat_colors * 255).astype(np.uint8)

        # 1) Main Surface Mesh
        self.plotter.add_mesh(
            grid,
            scalars="colors",
            rgb=True,
            show_edges=False,
            smooth_shading=True,
            specular=0.5,
            diffuse=0.9,
            opacity=0.92,
            lighting=True
        )

        # 2) Glowing Neon Anomalies Scatter
        # anom_win is (W, A) boolean mask
        # transpose matched Z shape -> (A, W)
        anom_x, anom_y = np.where(anom_win.T)
        if len(anom_x) > 0:
            anom_z = Z[anom_x, anom_y]
            
            # Map Alpha values to cmap for particles exactly like Matplotlib
            anom_alpha = alpha_win.T[anom_x, anom_y]
            
            # Map original Y index (0..3) to the new interpolated Y space
            mapped_y_coord = anom_x * ((target_A - 1) / max(A - 1, 1)) * (float(W) / target_A)
            
            points = np.column_stack((anom_y, mapped_y_coord, anom_z)).astype(np.float32)  
            cloud = pv.PolyData(points)
            cloud.point_data["intensity"] = anom_alpha
            
            # Outer halo
            self.plotter.add_mesh(
                cloud,
                render_points_as_spheres=True,
                point_size=18.0,
                scalars="intensity",
                cmap=NEON_ALPHA_CMAP,
                clim=[-1, 1],
                opacity=0.20,
                lighting=False
            )
            # Inner bright core
            self.plotter.add_mesh(
                cloud,
                render_points_as_spheres=True,
                point_size=6.0,
                scalars="intensity",
                cmap=NEON_ALPHA_CMAP,
                clim=[-1, 1],
                opacity=0.99,
                lighting=True
            )

        # Add 3D Bounding box and grid axes so it looks like the mathematical reference
        self.plotter.show_bounds(
            grid='front', 
            location='outer', 
            all_edges=False, 
            color='#6688aa',
            axes_ranges=[0, 90, 0, 4, 0, 1], # logical ticks instead of projected coordinates
            fmt="%.1f"
        )

        # Matplotlib used elev=45, azim=225. 
        # By pulling distance * 2.2 we isolate the 3D topology 
        if self.first_render:
            bounds = grid.bounds
            center = ((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2)
            distance = max(bounds[1]-bounds[0], bounds[3]-bounds[2]) * 2.2
            
            import math
            elev = math.radians(35)
            azim = math.radians(225)
            
            cx = center[0] + distance * math.cos(elev) * math.sin(azim)
            cy = center[1] + distance * math.cos(elev) * math.cos(azim)
            cz = center[2] + distance * math.sin(elev)
            
            self.plotter.camera_position = [(cx, cy, cz), center, (0, 0, 1)]
            self.first_render = False

        # Read back the buffer from the GPU
        img = self.plotter.screenshot(transparent_background=True, return_img=True)
        return img  # Returns (H, W, 4) NumPy array RGBA
