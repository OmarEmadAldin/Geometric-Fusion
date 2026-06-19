# import cv2
# import sys
# import os
# print(os.path.exists(r"F:\Omar 3amora\حنكشة projects\Geometric Fusion\Data\2011_09_26_drive_0001_sync\image_02\data\0000000000.png"))
# img = cv2.imread(r"F:\Omar 3amora\حنكشة projects\Geometric Fusion\Data\2011_09_26_drive_0001_sync\image_02\data\0000000000.png")
# if img is None:
#     raise IOError(f"Could not read image")
# while cv2.waitKey(1) != 27:  # Wait for 'Esc' key to exit
#     cv2.imshow("Image", img)

import cv2
import numpy as np

path = r"F:\Omar 3amora\حنكشة projects\Geometric Fusion\Data\2011_09_26_drive_0001_sync\image_02\data\0000000000.png"

img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

print(img is None)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()