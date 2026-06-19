import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works on headless servers)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List , Optional

def _depth_to_colors(depth: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Map depth values to RGBA colours using a matplotlib colormap."""
    d_min, d_max = depth.min(), depth.max()
    normalised = (depth - d_min) / (d_max - d_min + 1e-6)
    cmap = plt.get_cmap(colormap)
    return cmap(normalised)   # (N, 4) RGBA


def save_projection_all(
    image_bgr: np.ndarray,
    pts_2d: np.ndarray,
    depth: np.ndarray,
    output_path: str,
    point_size: int = 2,
    colormap: str = "jet",
) -> None:

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    colors = _depth_to_colors(depth, colormap)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5), dpi=120)
    ax.imshow(image_rgb)
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1],
               c=colors, s=point_size, linewidths=0)

    # Colourbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(colormap),
        norm=plt.Normalize(vmin=depth.min(), vmax=depth.max())
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Depth (m)", fontsize=9)

    ax.set_title(f"LiDAR projection — {len(pts_2d):,} visible points", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_projection_boxes(
    image_bgr: np.ndarray,
    pts_2d: np.ndarray,
    depth: np.ndarray,
    results: List[dict],
    output_path: str,
    bbox_colors: Optional[List[str]] = None,
    point_size: int = 2,
    colormap: str = "jet",
) -> None:

    if bbox_colors is None:
        bbox_colors = ["lime", "cyan", "magenta", "yellow", "orange", "red"]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    base_colors = _depth_to_colors(depth, colormap)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5), dpi=120)
    ax.imshow(image_rgb)

    # All points (muted, small)
    ax.scatter(pts_2d[:, 0], pts_2d[:, 1],
               c=base_colors, s=point_size, linewidths=0, alpha=0.5)

    # Per-object highlights
    for i, res in enumerate(results):
        c = bbox_colors[i % len(bbox_colors)]

        # Bounding box rectangle
        x1, y1, x2, y2 = res["x1"], res["y1"], res["x2"], res["y2"]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=c, facecolor="none"
        )
        ax.add_patch(rect)

        # Highlighted points
        if res["count"] > 0:
            ax.scatter(
                res["pts_2d"][:, 0], res["pts_2d"][:, 1],
                c=c, s=point_size * 4, zorder=5, linewidths=0,
                label=f"{res['label']}  {res['count']} pts  {res['min_depth']:.1f}m min"
            )

        # Label text above box
        label_txt = (
            f"{res['label']}  "
            f"avg={res['avg_depth']:.1f}m  "
            f"min={res['min_depth']:.1f}m  "
            f"pts={res['count']}"
        )
        ax.text(
            x1, max(y1 - 5, 10), label_txt,
            color=c, fontsize=7, weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, ec="none")
        )

    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    ax.set_title("LiDAR–Camera geometric fusion", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def draw_points_on_image_cv2(
    image_bgr: np.ndarray,
    pts_2d: np.ndarray,
    depth: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    radius: int = 2,
) -> np.ndarray:

    img = image_bgr.copy()
    d_min, d_max = depth.min(), depth.max()
    norm = ((depth - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(norm.reshape(-1, 1), colormap).reshape(-1, 3)

    for (u, v), color in zip(pts_2d.astype(int), color_map):
        cv2.circle(img, (u, v), radius, color.tolist(), -1)

    return img
