import torch
import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

def pose_to_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    t = np.array([x, y, z])
    rot = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    R_matrix = rot.as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_matrix
    T[:3, 3] = t
    return T # Returns T_W_L

def calculate_pose_error(pred_extrinsic_3x4, gt_extrinsic_4x4):
    gt = np.array(gt_extrinsic_4x4)
    pred = pred_extrinsic_3x4.detach().cpu().numpy()
    R_pred = pred[:3, :3]
    t_pred = pred[:3, 3]
    R_gt = gt[:3, :3]
    t_gt = gt[:3, 3]
    translation_error = np.linalg.norm(t_pred - t_gt)
    r_rel = R_pred @ R_gt.T
    trace = np.clip(np.trace(r_rel), -1.0, 3.0)
    rotation_error_rad = np.arccos((trace - 1.0) / 2.0)
    rotation_error_deg = np.degrees(rotation_error_rad)
    return translation_error, rotation_error_deg

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print("Loading model...")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
print("Model loaded.")

base_path = "dataset/v2x_vit"
mode = "train"
scenario = "2021_08_18_21_38_28"
vehicle_ids = ["8786"]
timestamps = ["000068", "000070"]
cameras = ["camera0", "camera1", "camera2", "camera3"]
camera_enabled = {
    "camera0": True,
    "camera1": True,
    "camera2": True,
    "camera3": True
}

mode_path = os.path.join(base_path, mode)

if scenario is None:
    scenario_folders = [f for f in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, f))]
    scenario = scenario_folders[0]

scenario_path = os.path.join(mode_path, scenario)

image_names = []
gt_extrinsics_ordered = []
camera_names_ordered = []
yaml_data_cache = {}
lidar_world_to_lidar_cache = {}

for timestamp in timestamps:
    for vehicle_id in vehicle_ids:
        yaml_key = (vehicle_id, timestamp)
        if yaml_key not in yaml_data_cache:
             yaml_path = os.path.join(scenario_path, vehicle_id, f"{timestamp}.yaml")
             with open(yaml_path, 'r') as f:
                 yaml_content = yaml.safe_load(f)
             yaml_data_cache[yaml_key] = yaml_content
             lidar_pose_6d = yaml_content['lidar_pose']
             M_lidar_pose_W_L = pose_to_matrix(lidar_pose_6d) # T_W_L
             T_L_W = np.linalg.inv(M_lidar_pose_W_L) # T_L_W = inv(T_W_L)
             lidar_world_to_lidar_cache[yaml_key] = T_L_W

        current_vehicle_yaml = yaml_data_cache[yaml_key]
        T_L_W = lidar_world_to_lidar_cache[yaml_key]
        vehicle_path = os.path.join(scenario_path, vehicle_id)

        for camera in cameras:
            if camera_enabled.get(camera, False):
                image_path = os.path.join(vehicle_path, f"{timestamp}_{camera}.png")
                image_names.append(image_path)
                camera_names_ordered.append(f"{vehicle_id}_{timestamp}_{camera}")

                final_gt_C_W = None
                if current_vehicle_yaml and camera in current_vehicle_yaml:
                     if 'extrinsic' in current_vehicle_yaml[camera]:
                          M_yaml_extrinsic_L_C = np.array(current_vehicle_yaml[camera]['extrinsic']) # T_L_C
                          T_C_L = np.linalg.inv(M_yaml_extrinsic_L_C) # T_C_L = inv(T_L_C)
                          final_gt_C_W = T_C_L @ T_L_W # GT T_C_W = T_C_L * T_L_W

                gt_extrinsics_ordered.append(final_gt_C_W)


print(f"Processing {len(image_names)} images from {len(timestamps)} timestamps simultaneously...")
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images_batch = images[None]
        aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        pred_extrinsic, pred_intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])


print(f"\nComparison Results (Processed Simultaneously, using lidar_pose)")
all_errors = {'trans': [], 'rot': []}
total_images = pred_extrinsic.shape[1]

for i in range(total_images):
     camera_id_name = camera_names_ordered[i]
     pred_matrix_3x4 = pred_extrinsic[0, i]
     gt_matrix_4x4 = gt_extrinsics_ordered[i]

     print(f"  Camera: {camera_id_name}")
     trans_err, rot_err = calculate_pose_error(pred_matrix_3x4, gt_matrix_4x4)
     print(f"    Translation Error: {trans_err:.4f}")
     print(f"    Rotation Error (deg): {rot_err:.4f}")
     all_errors['trans'].append(trans_err)
     all_errors['rot'].append(rot_err)


overall_avg_trans = np.mean(all_errors['trans'])
overall_avg_rot = np.mean(all_errors['rot'])
print("\n--- Overall Average Errors (Simultaneous Processing, using lidar_pose) ---")
print(f"Overall Avg Translation Error: {overall_avg_trans:.4f}")
print(f"Overall Avg Rotation Error (deg): {overall_avg_rot:.4f}")

print("\n--- Finished processing ---")