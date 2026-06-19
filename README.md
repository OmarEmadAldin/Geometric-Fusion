# LiDAR–Camera Geometric Fusion

Single-frame projection of 3D LiDAR point clouds onto a 2D camera image,
followed by per-object cropping using 2D bounding boxes.

The goal is to **project every LiDAR point onto the camera image** using calibration matrices, then **crop the points that fall inside a detected 2D bounding box** — giving you both the visual appearance and the 3D geometry of each object in the scene.


This is called **geometric sensor fusion** because it uses pure rigid-body geometry (rotation + translation + perspective projection) with no probabilistic estimation or time component.

---

## What this project does

1. Loads a KITTI LiDAR scan (`.bin`) and camera image (`.png`)
2. Reads calibration matrices (extrinsic R, t and intrinsic K)
3. Transforms every LiDAR point from LiDAR frame → Camera frame (extrinsic)
4. Filters out behind-camera points (Z ≤ 0)
5. Projects remaining points onto the image plane (intrinsic)
6. Filters out points that land outside the image boundary
7. Crops the points that fall inside each 2D bounding box (from a detector or manual input)

---

## The math, step by step
 - Step 1 — Extrinsic transform (LiDAR → Camera 3D frame):

        P_cam = R · P_lidar + t
        
    Each LiDAR point [X, Y, Z] becomes a point in the camera's coordinate system.

- Step 2 — Depth filter:
Keep only points where P_cam[2] > 0 (in front of the camera). Points behind the camera would project to nonsensical pixel coordinates.

- Step 3 — Intrinsic projection (Camera 3D → Image 2D):

        u = fx * (X_cam / Z_cam) + cx
        v = fy * (Y_cam / Z_cam) + cy

This gives you pixel coordinates (u, v) for each LiDAR point.

- Step 4 — Bounding box crop:
For each detection [x1, y1, x2, y2], keep only the points where:

        x1 ≤ u ≤ x2  AND  y1 ≤ v ≤ y2

The surviving 3D points are the ones corresponding to that detected object.

---

## Folder structure

```
lidar_camera_fusion/
│
├── README.md                   ← you are here
│
├── requirements.txt            ← Python dependencies
│
├── config.py                   ← all paths and tunable parameters
│
├── run_fusion.py               ← main entry point (runs everything)
│
├── utils/
│   ├── __init__.py
│   ├── calib_loader.py         ← parses KITTI calibration files
│   ├── point_cloud.py          ← loads and filters .bin point clouds
│   ├── projection.py           ← extrinsic transform + intrinsic projection
│   ├── bbox_crop.py            ← crops points by 2D bounding box
│   └── visualise.py            ← draws projected points + boxes on image
│
└── KITTI_calib_data/
│        ├── calib_velo_to_cam.txt   ← extrinsic calibration (R, t)
│        └── calib_cam_to_cam.txt    ← intrinsic calibration (K, distortion)
│
│
└── Data/2011_09_26_drive_0001_sync
     ├── image_00
     ├── image_01         
     ├── image_02          ← The folder we used
     ├── image_03          
     ├── oxts            
     └── velodyne_points
```

---
## Data download

You need exactly **two files** (~462 MB total):

| File | Size | URL |
|---|---|---|
| `2011_09_26_calib.zip` | 4 KB | https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip |
| `2011_09_26_drive_0001_sync.zip` | 458 MB | https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip |



### Config Tunable parameters

| Parameter | Default | Effect |
|---|---|---|
| `FRAME_NUMBER` | `0` | Which frame to process (0–107) |
| `BBOX_SOURCE` | `"yolo"` | `"yolo"` or `"manual"` |
| `MIN_DEPTH` | `0.5 m` | Drop LiDAR points closer than this |
| `MAX_DEPTH` | `80.0 m` | Drop LiDAR points beyond this |
| `POINT_SIZE` | `2` | Dot size in output images |
| `COLORMAP` | `"jet"` | Depth colour scheme (jet/plasma/viridis/turbo) |
| `BBOX_PADDING` | `0 px` | Expand each bounding box by N pixels |

---

## Pipeline stages explained

### Stage 1 — Load inputs
- Load `.bin` file as float32 array, shape (N, 4) → drop intensity → (N, 3)
- Load `.png` image with OpenCV
- Parse calibration `.txt` files into matrices

### Stage 2 — Extrinsic transform  (LiDAR frame → Camera frame)
```
P_cam = R_ext @ P_lidar + t_ext
```
`R_ext` and `t_ext` come from `calib_velo_to_cam.txt`.
After this step, Z in `P_cam` is the metric depth (distance from camera in metres).

### Stage 3 — Depth filter
```
keep only points where Z > 0
```
Points with Z ≤ 0 are behind the camera. Projecting them gives mirrored, wrong pixel coordinates.

### Stage 4 — Intrinsic projection  (Camera frame → Image plane)
```
u = fx * (X / Z) + cx      ← pixel column
v = fy * (Y / Z) + cy      ← pixel row
```
`fx, fy, cx, cy` come from `K_02` in `calib_cam_to_cam.txt`.

### Stage 5 — Image boundary filter
```
keep only points where  0 ≤ u < W  and  0 ≤ v < H
```
Removes projections that fall outside the image frame.

### Stage 6 — Bounding box crop
```
keep points where  x1 ≤ u ≤ x2  and  y1 ≤ v ≤ y2
```
For each 2D detection, selects the LiDAR points whose projected pixel falls inside the box. These are the 3D points belonging to that object.

--- 

## Why not Kalman filter here?

This pipeline is **geometric fusion** — it works on one frame at a time with no time component:

```
One frame  →  geometric fusion  →  "car is 15m away, here are its 3D points"
```

A **Kalman filter** operates across frames over time:

Frame 0: car at 15.2m  ─┐
Frame 1: car at 14.8m  ─┤  →  Kalman  →  track: car moving at -0.7m/frame
Frame 2: car at 14.1m  ─┤              →  predicted at 13.4m next frame
Frame 3: (occluded)    ─┤              →  still predicted even with no detection
Frame 4: car at 13.5m  ─┘

## Result

<p align="center">
  <img src="output/frame_0000/projection_all.png" width="45%" />
  <img src="output/frame_0000/projection_boxes.png" width="45%" />
</p>

---

<p align="center">
  <img src="projection_animation.gif" width="75%" />
  <img src="projection_boxes_animation.gif" width="75%" />
</p>
