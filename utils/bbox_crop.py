from __future__ import annotations
import numpy as np
import typing

def crop_by_bbox( pts_2d: np.ndarray, P_cam: np.ndarray,P_lidar: np.ndarray,depth: np.ndarray,bbox: dict,padding: int = 0,) -> dict:
    """
    Parameters
    ----------
    pts_2d  : (N, 2)   pixel coords of all visible projected points
    P_cam   : (N, 3)   camera-frame XYZ
    P_lidar : (N, 3)   LiDAR-frame XYZ
    depth   : (N,)     metric depth
    bbox    : dict with keys x1, y1, x2, y2, label
    padding : int      expand box by this many pixels on each side
    """
    x1 = bbox["x1"] - padding
    y1 = bbox["y1"] - padding
    x2 = bbox["x2"] + padding
    y2 = bbox["y2"] + padding

    u, v = pts_2d[:, 0], pts_2d[:, 1]
    mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)

    cropped_depth = depth[mask]
    avg_d = float(cropped_depth.mean()) if mask.sum() > 0 else 0.0
    min_d = float(cropped_depth.min()) if mask.sum() > 0 else 0.0

    return {
        "pts_2d":    pts_2d[mask],
        "P_cam":     P_cam[mask],
        "P_lidar":   P_lidar[mask],
        "depth":     cropped_depth,
        "label":     bbox.get("label", "object"),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "count":     int(mask.sum()),
        "avg_depth": avg_d,
        "min_depth": min_d,
    }


def crop_all(pts_2d: np.ndarray,P_cam: np.ndarray,P_lidar: np.ndarray,depth: np.ndarray,bboxes: list[dict],padding: int = 0,) -> list[dict]:
    return [
        crop_by_bbox(pts_2d, P_cam, P_lidar, depth, bb, padding)
        for bb in bboxes
    ]


def load_manual_bboxes(manual_list: list[dict]) -> list[dict]: return manual_list

def load_kitti_labels(label_path: str) -> list[dict]:
    
    bboxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            label       = parts[0]
            if label == "DontCare":
                continue
            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            bboxes.append({
                "label": label,
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
            })
    return bboxes


def load_yolo_bboxes(image_path: str, conf_thresh: float = 0.4) -> list[dict]:
   
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not installed. Run: pip install ultralytics"
        )
    model = YOLO("yolov8n.pt")   # downloads automatically on first run
    results = model(image_path, verbose=False)[0]

    bboxes = []
    for box in results.boxes:
        conf = float(box.conf)
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = results.names[int(box.cls)]
        bboxes.append({
            "label": label,
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "conf": round(conf, 3),
        })
    return bboxes


def get_bboxes(source: str, config) -> list[dict]:
    
    if source == "manual":
        return load_manual_bboxes(config.MANUAL_BBOXES)

    elif source == "kitti_labels":
        return load_kitti_labels(config.LABEL_PATH)

    elif source == "yolo":
        return load_yolo_bboxes(config.IMAGE_PATH)

    else:
        raise ValueError(
            f"Unknown BBOX_SOURCE '{source}'. Choose from: 'manual', 'kitti_labels', 'yolo'")
