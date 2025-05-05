import torch
import os
import yaml
import numpy as np

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

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
# vehicle_ids = ["8786", "8795"]
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

all_timestamp_errors = []

for timestamp in timestamps:
    print(f"\n--- Processing Timestamp: {timestamp} ---")

    image_names = []
    gt_extrinsics_ordered = []
    camera_names_ordered = []

    yaml_data_map = {}
    for vehicle_id in vehicle_ids:
         yaml_path = os.path.join(scenario_path, vehicle_id, f"{timestamp}.yaml")
         with open(yaml_path, 'r') as f:
             yaml_data = yaml.safe_load(f)
         yaml_data_map[vehicle_id] = yaml_data


    for vehicle_id in vehicle_ids:
        vehicle_path = os.path.join(scenario_path, vehicle_id)
        current_vehicle_yaml = yaml_data_map.get(vehicle_id)

        for camera in cameras:
            if camera_enabled.get(camera, False):
                image_path = os.path.join(vehicle_path, f"{timestamp}_{camera}.png")
                image_names.append(image_path)
                camera_names_ordered.append(f"{vehicle_id}_{camera}")

                gt_extrinsic = None
                if current_vehicle_yaml and camera in current_vehicle_yaml:
                     if 'extrinsic' in current_vehicle_yaml[camera]:
                          gt_extrinsic = current_vehicle_yaml[camera]['extrinsic']

                gt_extrinsics_ordered.append(gt_extrinsic)


    print(f"Processing {len(image_names)} images for timestamp {timestamp}")
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            pred_extrinsic, pred_intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    print(f"\nComparison Results for Timestamp: {timestamp}")

    num_predictions = pred_extrinsic.shape[1]
    timestamp_errors = {'trans': [], 'rot': []}
    for i in range(num_predictions):
         camera_id_name = camera_names_ordered[i]
         pred_matrix_3x4 = pred_extrinsic[0, i]
         gt_matrix_4x4 = gt_extrinsics_ordered[i]

         print(f"  Camera: {camera_id_name}")
         trans_err, rot_err = calculate_pose_error(pred_matrix_3x4, gt_matrix_4x4)
         print(f"    Translation Error: {trans_err:.4f}")
         print(f"    Rotation Error (deg): {rot_err:.4f}")
         timestamp_errors['trans'].append(trans_err)
         timestamp_errors['rot'].append(rot_err)

    avg_trans_error = np.mean(timestamp_errors['trans'])
    avg_rot_error = np.mean(timestamp_errors['rot'])
    print(f"\n  Average Errors for Timestamp {timestamp}:")
    print(f"    Avg Translation Error: {avg_trans_error:.4f}")
    print(f"    Avg Rotation Error (deg): {avg_rot_error:.4f}")
    all_timestamp_errors.append({'timestamp': timestamp, 'avg_trans': avg_trans_error, 'avg_rot': avg_rot_error})


print("\n--- Finished processing all timestamps ---")

print("\n--- Overall Average Errors Across Timestamps ---")
overall_avg_trans = np.mean([err['avg_trans'] for err in all_timestamp_errors])
overall_avg_rot = np.mean([err['avg_rot'] for err in all_timestamp_errors])
print(f"Overall Avg Translation Error: {overall_avg_trans:.4f}")
print(f"Overall Avg Rotation Error (deg): {overall_avg_rot:.4f}")