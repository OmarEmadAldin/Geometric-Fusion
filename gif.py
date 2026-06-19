from pathlib import Path
import imageio.v2 as imageio
import cv2

output_dir = Path("output")

image_files = sorted(
    [d / "projection_boxes.png"
     for d in output_dir.glob("frame_*")
     if (d / "projection_boxes.png").exists()]
)

# Use first image size as reference
first = imageio.imread(image_files[0])
height, width = first.shape[:2]

images = []

for img_path in image_files:
    img = imageio.imread(img_path)

    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))

    images.append(img)

imageio.mimsave(
    "projection_boxes_animation.gif",
    images,
    duration=0.1
)

print(f"Saved GIF with {len(images)} frames")
