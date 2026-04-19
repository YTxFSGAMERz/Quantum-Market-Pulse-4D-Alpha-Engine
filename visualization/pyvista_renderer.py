import numpy as np
import pyvista as pv
from visualization.surface_renderer import _build_facecolors
from utils.helpers import NEON_ALPHA_CMAP

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

        # Build Mesh Grid Data
        W, A = E_win.shape
        x = np.arange(W)
        y = np.arange(A) * 2  # spacing modifier if needed, matching matplotlib scale
        X, Y = np.meshgrid(x, y) 
        Z = E_win.T

        # Create structured grid
        grid = pv.StructuredGrid(X, Y, Z)
        
        # We must re-shape facecolors to point scalars
        # Matplotlib uses a per-vertex approach
        facecolors = _build_facecolors(E_win, alpha_win) # (A, W, 4)
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
            
            # We map -1..1 to cmap. In PyVista, scalars to colormap is standard
            points = np.column_stack((anom_y, anom_x * 2, anom_z))  
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

        # Matplotlib used elev=45, azim=225. 
        # In PyVista, camera position viewing origin (W/2, A/2, Z mean)
        if self.first_render:
            bounds = grid.bounds
            center = ((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2)
            # Distance from center
            distance = max(bounds[1]-bounds[0], bounds[3]-bounds[2]) * 1.5
            
            import math
            elev = math.radians(45)
            azim = math.radians(225)
            
            cx = center[0] + distance * math.cos(elev) * math.sin(azim)
            cy = center[1] + distance * math.cos(elev) * math.cos(azim)
            cz = center[2] + distance * math.sin(elev)
            
            self.plotter.camera_position = [(cx, cy, cz), center, (0, 0, 1)]
            self.first_render = False

        # Read back the buffer from the GPU
        img = self.plotter.screenshot(transparent_background=True, return_img=True)
        return img  # Returns (H, W, 4) NumPy array RGBA
