import numpy as np

def load_bin(bin_path: str) -> np.ndarray:
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def preprocess(
    points: np.ndarray,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
    remove_ego: bool = True,
    ego_radius: float = 1.5,
) -> np.ndarray:
    
    xyz = points[:, :3]

    # Forward distance in the Velodyne frame ≈ X component
    forward_dist = xyz[:, 0]
    mask = (forward_dist > min_depth) & (forward_dist < max_depth)

    # Remove points within ego-vehicle radius (noise from roof/hood)
    if remove_ego:
        r = np.linalg.norm(xyz, axis=1)
        mask &= (r > ego_radius)

    return xyz[mask]


def load_and_preprocess(
    bin_path: str,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
) -> np.ndarray:
   
    raw = load_bin(bin_path)
    return preprocess(raw, min_depth=min_depth, max_depth=max_depth)
