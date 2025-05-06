import torch
import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

def matrix_to_tum_line(matrix_4x4, timestamp_id):
    t = matrix_4x4[:3, 3]
    R_mat = matrix_4x4[:3, :3]
    q = R.from_matrix(R_mat).as_quat()
    return f"{timestamp_id} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print("Loading model...")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
print("Model loaded.")

base_path = "dataset/v2x_vit"
mode = "train"
scenario = "2021_08_22_07_24_12"
ego_vehicle_id = "5274" # Define the ego vehicle ID
vehicle_ids = ["5274", "5292"] # All vehicles used for input
timestamps = ["000078"]
cameras = ["camera0", "camera1", "camera2", "camera3"]
camera_enabled = {
    "camera0": True,
    "camera1": True,
    "camera2": True,
    "camera3": True
}

num_vehicles = len(vehicle_ids)
num_timestamps = len(timestamps)

mode_path = os.path.join(base_path, mode)

if scenario is None:
    scenario_folders = [f for f in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, f))]
    scenario = scenario_folders[0]

scenario_path = os.path.join(mode_path, scenario)

image_names = []
gt_extrinsics_ordered = []
camera_names_ordered = []
yaml_data_cache = {}

for timestamp in timestamps:
    for vehicle_id in vehicle_ids:
        yaml_key = (vehicle_id, timestamp)
        if yaml_key not in yaml_data_cache:
             yaml_path = os.path.join(scenario_path, vehicle_id, f"{timestamp}.yaml")
             with open(yaml_path, 'r') as f:
                 yaml_content = yaml.safe_load(f)
             yaml_data_cache[yaml_key] = yaml_content

        current_vehicle_yaml = yaml_data_cache[yaml_key]
        vehicle_path = os.path.join(scenario_path, vehicle_id)

        for camera in cameras:
            if camera_enabled.get(camera, False):
                image_path = os.path.join(vehicle_path, f"{timestamp}_{camera}.png")
                image_names.append(image_path)
                camera_names_ordered.append(f"{vehicle_id}_{timestamp}_{camera}")

                gt_extrinsic = None
                if current_vehicle_yaml and camera in current_vehicle_yaml:
                     if 'extrinsic' in current_vehicle_yaml[camera]:
                          gt_extrinsic = np.array(current_vehicle_yaml[camera]['extrinsic'])
                gt_extrinsics_ordered.append(gt_extrinsic)


print(f"Processing {len(image_names)} images for {num_vehicles} vehicles (Ego: {ego_vehicle_id}) and {num_timestamps} timestamps simultaneously...")
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images_batch = images[None]
        aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        pred_extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

# --- Prepare data for evo (FILTERED for ego vehicle) ---
gt_tum_lines = []
pred_tum_lines = []
total_images = pred_extrinsic.shape[1]

for i in range(total_images):
    gt_matrix_4x4 = gt_extrinsics_ordered[i]
    pred_matrix_3x4_tensor = pred_extrinsic[0, i]
    full_camera_name = camera_names_ordered[i] # Format: "vehicle_timestamp_camera"

    # --- Filtering Step ---
    # Extract vehicle ID from the name
    vehicle_id_from_name = full_camera_name.split('_')[0]

    # Process only if GT exists AND it's the ego vehicle
    if gt_matrix_4x4 is not None and vehicle_id_from_name == ego_vehicle_id:
        pred_matrix_4x4 = np.eye(4)
        pred_matrix_4x4[:3, :] = pred_matrix_3x4_tensor.detach().cpu().numpy()
        timestamp_id = i # Using original index i for TUM timestamp ID
        gt_line = matrix_to_tum_line(gt_matrix_4x4, timestamp_id)
        pred_line = matrix_to_tum_line(pred_matrix_4x4, timestamp_id)
        gt_tum_lines.append(gt_line)
        pred_tum_lines.append(pred_line)

# --- File Saving Logic (Directory structure reflects INPUT conditions) ---
base_output_dir = "evo_input"
scenario_output_dir = os.path.join(base_output_dir, scenario)
vehicle_folder_name = f"{num_vehicles}_vehicle" if num_vehicles == 1 else f"{num_vehicles}_vehicles"
timestamp_folder_name = f"{num_timestamps}_timestamp" if num_timestamps == 1 else f"{num_timestamps}_timestamps"
final_output_dir = os.path.join(scenario_output_dir, vehicle_folder_name, timestamp_folder_name)

os.makedirs(final_output_dir, exist_ok=True)

gt_filename = os.path.join(final_output_dir, "gt_poses.txt")
pred_filename = os.path.join(final_output_dir, "pred_poses.txt")

with open(gt_filename, 'w') as f:
    for line in gt_tum_lines:
        f.write(line + '\n')
# Print the count of poses actually saved (ego vehicle only)
print(f"Ground truth poses (Ego vehicle only) saved to: {gt_filename} ({len(gt_tum_lines)} poses)")

with open(pred_filename, 'w') as f:
    for line in pred_tum_lines:
        f.write(line + '\n')
# Print the count of poses actually saved (ego vehicle only)
print(f"Predicted poses (Ego vehicle only) saved to: {pred_filename} ({len(pred_tum_lines)} poses)")

print("\n--- Finished processing ---")