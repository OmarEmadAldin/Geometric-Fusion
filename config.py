import os
DATASET_FORMAT = "raw"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

_drive = os.path.join(PROJECT_ROOT, "Data", "2011_09_26_drive_0001_sync")
_calib = os.path.join(PROJECT_ROOT, "KITTI_calib_data")

FRAME_NUMBER = 0

# Derived paths — do not edit these
IMAGE_PATH = os.path.join(_drive, "image_02", "data", f"{FRAME_NUMBER:010d}.png")
LIDAR_PATH = os.path.join(_drive, "velodyne_points", "data", f"{FRAME_NUMBER:010d}.bin")
CALIB_VELO = os.path.join(_calib, "calib_velo_to_cam.txt")
CALIB_CAM  = os.path.join(_calib, "calib_cam_to_cam.txt")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
BBOX_SOURCE = "yolo"   # auto-detects objects — no manual boxes needed
MANUAL_BBOXES = [
    {"x1": 550, "y1": 160, "x2": 720, "y2": 270, "label": "car"},
]
MIN_DEPTH = 0.5
MAX_DEPTH = 80.0

POINT_SIZE  = 2
COLORMAP    = "jet"
BBOX_COLORS = ["lime", "cyan", "magenta", "yellow", "orange", "red"]
BBOX_PADDING = 0