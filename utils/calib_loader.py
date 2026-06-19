'''
File created to ge the matrices for calib data and calib params
'''
import numpy as np
def load_split_calib(velo_to_cam_path: str, cam_to_cam_path: str) -> dict:

    def parse_file(path):
        data = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, vals = line.split(":", 1)
                nums = []
                for v in vals.strip().split():
                    try:
                        nums.append(float(v))
                    except ValueError:
                        pass   # skip non-numeric tokens like dates
                if nums:
                    data[key.strip()] = np.array(nums)
        return data

    ext_data = parse_file(velo_to_cam_path)
    int_data = parse_file(cam_to_cam_path)

    R = ext_data["R"].reshape(3, 3)
    t = ext_data["T"].reshape(3)

    K = int_data["K_02"].reshape(3, 3)
    dist_coeffs = int_data["D_02"].flatten()
    R0_rect= int_data.get("R_rect_02", np.eye(3)).reshape(3, 3)

    R_ext = R0_rect @ R
    t_ext = R0_rect @ t

    return {
        "K":           K,
        "R_ext":       R_ext,
        "t_ext":       t_ext,
        "dist_coeffs": dist_coeffs,
        "R0_rect":     R0_rect,
    }