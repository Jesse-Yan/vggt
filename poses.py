import torch
import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

def matrix_to_tum_line(matrix_4x4, timestamp_id):
    # Assumes input is T_W_C (Camera-to-World)
    t = matrix_4x4[:3, 3]
    R_mat = matrix_4x4[:3, :3]
    q = R.from_matrix(R_mat).as_quat()
    return f"{timestamp_id} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print("Loading model...")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
print("Model loaded.")
if False:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_22_07_24_12"
    ego_vehicle_id = "5274"
    # vehicle_ids = ["5274"]
    vehicle_ids = ["5274", "5292"]
    base_timestamp = "000078" # Base timestamp for the ego vehicle

if False:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_23_13_10_47"
    ego_vehicle_id = "7694"
    # vehicle_ids = ["7694"]
    vehicle_ids = ["7694", "7703"]
    base_timestamp = "000098" # Base timestamp for the ego vehicle

if False:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_21_09_09_41"
    ego_vehicle_id = "9224"
    # vehicle_ids = ["9224"]
    vehicle_ids = ["9224", "9206"]
    base_timestamp = "000142" # Base timestamp for the ego vehicle

if False:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_22_09_43_53"
    ego_vehicle_id = "8323"
    # vehicle_ids = ["8323"]
    vehicle_ids = ["8323", "8332"]
    base_timestamp = "000119" # Base timestamp for the ego vehicle

if False:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_22_10_10_40"
    ego_vehicle_id = "8482"
    # vehicle_ids = ["8482"]
    vehicle_ids = ["8482", "8464"]
    base_timestamp = "000175" # Base timestamp for the ego vehicle

if False:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_23_12_13_48"
    ego_vehicle_id = "7365"
    # vehicle_ids = ["7365"]
    vehicle_ids = ["7365", "7356"]
    base_timestamp = "000074" # Base timestamp for the ego vehicle

if False:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_23_20_47_11"
    ego_vehicle_id = "409"
    # vehicle_ids = ["409"]
    vehicle_ids = ["409", "418"]
    base_timestamp = "000186" # Base timestamp for the ego vehicle

if False:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_23_22_31_01"
    ego_vehicle_id = "279"
    # vehicle_ids = ["279"]
    vehicle_ids = ["279", "252"]
    base_timestamp = "000256" # Base timestamp for the ego vehicle

if False:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_23_23_08_17"
    ego_vehicle_id = "565"
    # vehicle_ids = ["565"]
    vehicle_ids = ["565", "574"]
    base_timestamp = "000189" # Base timestamp for the ego vehicle

if False:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_24_09_25_42"
    ego_vehicle_id = "12963"
    # vehicle_ids = ["12963"]
    vehicle_ids = ["12963", "12954"]
    base_timestamp = "000134" # Base timestamp for the ego vehicle

if True:
    base_path = "dataset/v2x_vit"
    mode = "train"
    scenario = "2021_08_24_09_58_32"
    ego_vehicle_id = "13099"
    # vehicle_ids = ["13099"]
    vehicle_ids = ["13099", "13090"]
    base_timestamp = "000226" # Base timestamp for the ego vehicle

num_frames = 4

timestamps = []
for i in range(num_frames):
    # Generate timestamps based on the base timestamp
    new_timestamp = str(int(base_timestamp) + i * 2).zfill(6)  # Increment by 2 for each step
    timestamps.append(new_timestamp)
cameras = ["camera0", "camera1", "camera2", "camera3"]
camera_enabled = {
    "camera0": True,
    "camera1": False,
    "camera2": False,
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
gt_extrinsics_ordered = [] # Stores calculated GT T_W_C
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

                final_gt_W_C = None 
                if current_vehicle_yaml and camera in current_vehicle_yaml:
                     if 'extrinsic' in current_vehicle_yaml[camera]:
                          M_yaml_extrinsic_L_C = np.array(current_vehicle_yaml[camera]['extrinsic'])
                          final_gt_W_C = M_yaml_extrinsic_L_C

                gt_extrinsics_ordered.append(final_gt_W_C) # Append calculated T_W_C (or None)


print(f"Processing {len(image_names)} images for {num_vehicles} vehicles (Ego: {ego_vehicle_id}) and {num_timestamps} timestamps simultaneously...")
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images_batch = images[None]
        aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # pred_extrinsic is predicted T_W_C (Camera-to-World)
        pred_extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])


gt_tum_lines = []
pred_tum_lines = []
total_images = pred_extrinsic.shape[1]

for i in range(total_images):
    gt_matrix_4x4_W_C = gt_extrinsics_ordered[i] # Calculated GT T_W_C (4x4) or None
    pred_matrix_3x4_tensor_W_C = pred_extrinsic[0, i] # Predicted T_W_C (3x4)
    full_camera_name = camera_names_ordered[i]

    vehicle_id_from_name = full_camera_name.split('_')[0]

    # Filter for ego vehicle and valid GT
    if gt_matrix_4x4_W_C is not None and vehicle_id_from_name == ego_vehicle_id:
        # Convert prediction T_W_C to 4x4 numpy
        pred_matrix_4x4_W_C = np.eye(4)
        pred_matrix_4x4_W_C[:3, :] = pred_matrix_3x4_tensor_W_C.detach().cpu().numpy()

        timestamp_id = i

        # Convert both T_W_C matrices to TUM lines
        gt_line = matrix_to_tum_line(gt_matrix_4x4_W_C, timestamp_id)
        pred_line = matrix_to_tum_line(pred_matrix_4x4_W_C, timestamp_id)

        gt_tum_lines.append(gt_line)
        pred_tum_lines.append(pred_line)

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
print(f"Ground truth poses (Ego vehicle only) saved to: {gt_filename} ({len(gt_tum_lines)} poses)")

with open(pred_filename, 'w') as f:
    for line in pred_tum_lines:
        f.write(line + '\n')
print(f"Predicted poses (Ego vehicle only) saved to: {pred_filename} ({len(pred_tum_lines)} poses)")


print("\n--- Finished processing ---")