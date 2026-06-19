import os
import json
import numpy as np
import cv2
import config
from utils.calib_loader import load_split_calib
from utils.point_cloud  import load_and_preprocess
from utils.projection   import project_lidar_to_image
from utils.bbox_crop    import get_bboxes, crop_all
from utils.visualise    import save_projection_all, save_projection_boxes
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def main():
    print("Geometric Fusion: LiDAR + Camera  (KITTI dataset)".center(60, "="))
    print(f"data path is{config._drive}")
    print(f"calib path is{config._calib}")
    print(f"Images path is{config.IMAGE_PATH}")
    print(f"LiDAR path is{config.LIDAR_PATH}")

    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    for path, name in [
        (config.IMAGE_PATH, "Camera image"),
        (config.LIDAR_PATH, "LiDAR .bin"),
        (config.CALIB_VELO, "calib_velo_to_cam.txt"),
        (config.CALIB_CAM,  "calib_cam_to_cam.txt"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found:\n  {path}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print(f"\n[1] Loading inputs  (frame {config.FRAME_NUMBER:010d})")

    image = cv2.imdecode(np.fromfile(config.IMAGE_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Could not read image: {config.IMAGE_PATH}")
    H, W = image.shape[:2]

    points_lidar = load_and_preprocess(
        config.LIDAR_PATH,
        min_depth=config.MIN_DEPTH,
        max_depth=config.MAX_DEPTH,
    )
    print(f"LiDAR points : {len(points_lidar):,}  (after pre-filter)")

    calib = load_split_calib(config.CALIB_VELO, config.CALIB_CAM)
    print(f"Calibration  : loaded")
    print(f"fx={calib['K'][0,0]:.1f}  fy={calib['K'][1,1]:.1f}  " f"cx={calib['K'][0,2]:.1f}  cy={calib['K'][1,2]:.1f}")

    # ------------------------------------------------------------------
    # 2. Load bounding boxes
    # ------------------------------------------------------------------
    print(f"\n[2] Bounding boxes  (source: {config.BBOX_SOURCE})")
    bboxes = get_bboxes(config.BBOX_SOURCE, config)
    for i, bb in enumerate(bboxes):
        print(f"[{i}] {bb['label']:15s} "f"x1={bb['x1']} y1={bb['y1']} x2={bb['x2']} y2={bb['y2']}")

    # ------------------------------------------------------------------
    # 3. Project LiDAR → image plane
    # ------------------------------------------------------------------
    print("\n Projecting LiDAR points onto image plane")
    proj = project_lidar_to_image(
        points_lidar = points_lidar,
        R_ext        = calib["R_ext"],
        t_ext        = calib["t_ext"],
        K            = calib["K"],
        image_height = H,
        image_width  = W,
        dist_coeffs  = calib["dist_coeffs"],
        min_z        = 0.1,
    )
    pct = 100 * proj["n_visible"] / max(proj["n_input"], 1)
    print(f"    Input  : {proj['n_input']:,} pts")
    print(f"    Visible: {proj['n_visible']:,} pts  ({pct:.1f}% in image)")
    print(f"    Depth  : {proj['depth'].min():.1f}m – {proj['depth'].max():.1f}m")

    # ------------------------------------------------------------------
    # 4. Crop points per bounding box
    # ------------------------------------------------------------------
    print("\n Cropping by bounding box")

    results = crop_all(
        pts_2d  = proj["pts_2d"],
        P_cam   = proj["P_cam"],
        P_lidar = proj["P_lidar"],
        depth   = proj["depth"],
        bboxes  = bboxes,
        padding = config.BBOX_PADDING,
    )
    for i, res in enumerate(results):
        print(f"    [{i}] {res['label']:15s}  "
              f"pts={res['count']:5d}  "
              f"avg={res['avg_depth']:.2f}m  "
              f"min={res['min_depth']:.2f}m")

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    print(f"\n[5] Saving to: {config.OUTPUT_DIR}")

    save_projection_all(
        image_bgr   = image,
        pts_2d      = proj["pts_2d"],
        depth       = proj["depth"],
        output_path = os.path.join(config.OUTPUT_DIR, "projection_all.png"),
        point_size  = config.POINT_SIZE,
        colormap    = config.COLORMAP,
    )

    save_projection_boxes(
        image_bgr   = image,
        pts_2d      = proj["pts_2d"],
        depth       = proj["depth"],
        results     = results,
        output_path = os.path.join(config.OUTPUT_DIR, "projection_boxes.png"),
        bbox_colors = config.BBOX_COLORS,
        point_size  = config.POINT_SIZE,
        colormap    = config.COLORMAP,
    )

    print("\n" + "=" * 60)
    print("  Done! Check the output/ folder for results.")
    print("=" * 60)


if __name__ == "__main__":
    
    RUN_ALL = True
    if RUN_ALL: # using all frames leads us to using kalman filter but i can do it random in this code just for showing the res
        # Find all .bin files to know how many frames exist
        bin_files = sorted(glob.glob(
            os.path.join(config._drive, "velodyne_points", "data", "*.bin")
        ))
        total = len(bin_files)
        print(f"Processing all {total} frames...")

        for frame_num in range(total):
            print(f"\n{'='*60}")
            print(f"  Frame {frame_num + 1} of {total}")
            print(f"{'='*60}")

            config.FRAME_NUMBER = frame_num
            config.IMAGE_PATH = os.path.join(
                config._drive, "image_02", "data", f"{frame_num:010d}.png"
            )
            config.LIDAR_PATH = os.path.join(
                config._drive, "velodyne_points", "data", f"{frame_num:010d}.bin"
            )
            # Save outputs in a subfolder per frame
            config.OUTPUT_DIR = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "output", f"frame_{frame_num:04d}"
            )
            try:
                main()
            except Exception as e:
                print(f"  Frame {frame_num} failed: {e}")
                continue

        print(f"\nAll {total} frames done. Results in output/frame_XXXX/")
    else:
        main()