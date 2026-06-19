"""
The mathematical tansformation from 3D LiDAR points to 2D camera pixels consists of four stages
1. Extrinsic transform    P_cam = R_ext @ P_lidar + t_ext
2. Depth filter           keep Z > 0
3. Intrinsic projection   u = fx*(X/Z)+cx,  v = fy*(Y/Z)+cy
4. Image boundary filter  keep 0 ≤ u < W  and  0 ≤ v < H
All operations are vectorised NumPy — no Python loops over points. 34an el parallization
"""
import numpy as np
import cv2
from typing import Tuple , Optional
def transform_to_camera_frame(points_lidar: np.ndarray,R_ext: np.ndarray,t_ext: np.ndarray,) -> np.ndarray:
    
    #    Rigid-body transform from LiDAR frame to camera frame.
    #    P_cam = R_ext * P_lidar + t_ext 
    P_cam = points_lidar @ R_ext.T + t_ext
    return P_cam
# ---------------------------------------------------------------------------
# Step 2 — Depth filter
# ---------------------------------------------------------------------------
def filter_by_depth(P_cam: np.ndarray,points_lidar: np.ndarray,min_z: float = 0.1,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   # Remove points that are behind or at the camera plane (Z ≤ min_z).
    mask = P_cam[:, 2] > min_z
    return P_cam[mask], points_lidar[mask], P_cam[mask, 2]
# ---------------------------------------------------------------------------
# Step 3 — Intrinsic projection
# ---------------------------------------------------------------------------
def project_to_image(P_cam: np.ndarray,K: np.ndarray,dist_coeffs: Optional[np.ndarray] = None,) -> np.ndarray:
    """
    Project 3D camera-frame points to 2D pixel coordinates.
    Two modes:
    - dist_coeffs is None or all zeros → plain pinhole formula (fast)
    - dist_coeffs has non-zero values  → use cv2.projectPoints (handles radial + tangential distortion; use for raw / un-rectified images)

    """
    use_distortion = ( dist_coeffs is not None and not np.allclose(dist_coeffs, 0))

    if use_distortion:
        pts_2d, _ = cv2.projectPoints(
            P_cam.astype(np.float64).reshape(-1, 1, 3),
            np.zeros(3),   # rvec — already in camera frame
            np.zeros(3),   # tvec — already in camera frame
            K.astype(np.float64),
            dist_coeffs.astype(np.float64),
        )
        return pts_2d.reshape(-1, 2)

    # Plain pinhole
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.stack([u, v], axis=1)
# ---------------------------------------------------------------------------
# Step 4 — Image boundary filter
# ---------------------------------------------------------------------------
def filter_in_image(
    pts_2d: np.ndarray,
    P_cam: np.ndarray,
    points_lidar: np.ndarray,
    depth: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    u, v = pts_2d[:, 0], pts_2d[:, 1]
    mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    return pts_2d[mask], P_cam[mask], points_lidar[mask], depth[mask] # Filtered Array

# ---------------------------------------------------------------------------
# Full pipeline in one call
# ---------------------------------------------------------------------------

def project_lidar_to_image(
    points_lidar: np.ndarray,
    R_ext: np.ndarray,
    t_ext: np.ndarray,
    K: np.ndarray,
    image_height: int,
    image_width: int,
    dist_coeffs: Optional[np.ndarray] = None,
    min_z: float = 0.1,) -> dict:

    n_input = len(points_lidar)
    # Stage 1: extrinsic
    P_cam = transform_to_camera_frame(points_lidar, R_ext, t_ext)
    # Stage 2: depth filter
    P_cam, points_lidar, depth = filter_by_depth(P_cam, points_lidar, min_z)
    # Stage 3: intrinsic projection
    pts_2d = project_to_image(P_cam, K, dist_coeffs)
    # Stage 4: image boundary filter
    pts_2d, P_cam, points_lidar, depth = filter_in_image(
        pts_2d, P_cam, points_lidar, depth, image_width, image_height
    )
    return {
        "pts_2d":    pts_2d,
        "P_cam":     P_cam,
        "P_lidar":   points_lidar,
        "depth":     depth,
        "n_input":   n_input,
        "n_visible": len(pts_2d),
    }
