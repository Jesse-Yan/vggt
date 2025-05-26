import torch
import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

# --- SCENARIOS Configuration
from scenarios_config import SCENARIOS

# --- Helper Function ---
def matrix_to_tum_line(matrix_4x4, timestamp_id_tum):
    # Assumes input is T_W_C (Camera-to-World)
    t = matrix_4x4[:3, 3]
    R_mat = matrix_4x4[:3, :3]
    q = R.from_matrix(R_mat).as_quat()
    return f"{timestamp_id_tum} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}"

# --- Global Settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print("Loading model...")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
print("Model loaded.")

BASE_PATH_GLOBAL = "dataset/v2x_vit"
MODE_GLOBAL = "train"
CAMERAS_GLOBAL = ["camera0", "camera1", "camera2", "camera3"]
CAMERA_ENABLED_GLOBAL = {
    "camera0": True,
    "camera1": False,
    "camera2": False,
    "camera3": True
}
BASE_OUTPUT_DIR_GLOBAL = "evo_vggt"

# --- Main Processing Loop ---
for scenario_name, scenario_config in SCENARIOS.items():
    print(f"\n[SCENARIO]: {scenario_name}")
    # Example of accessing the new ref values
    # current_base_ts_ref = scenario_config["base_timestamp_ref"]
    # current_end_ts_ref = scenario_config["end_timestamp_ref"]
    # print(f"  Reference Timestamps: Base={current_base_ts_ref}, End={current_end_ts_ref}")
    print("-----------------------------------")

    ego_id_for_filter = scenario_config["ego_vehicle_id"]
    all_vehicles_in_scenario_config = scenario_config["vehicle_ids"]

    for timestamp_group_key, integer_timestamp_list in scenario_config["timestamps_data"].items():
        
        current_sequence_timestamps_str = [str(ts_int).zfill(6) for ts_int in integer_timestamp_list]
        num_frames_for_sequence = len(current_sequence_timestamps_str)
        
        print(f"  [TIMESTAMP_GROUP_ID]: {timestamp_group_key} (Sequence: {current_sequence_timestamps_str})")

        vehicle_run_cases = []
        vehicle_run_cases.append({
            "case_folder_suffix": "1_vehicle",
            "vehicles_to_process": [ego_id_for_filter]
        })
        if set(all_vehicles_in_scenario_config) != set([ego_id_for_filter]):
            vehicle_run_cases.append({
                "case_folder_suffix": f"{len(all_vehicles_in_scenario_config)}_vehicles",
                "vehicles_to_process": all_vehicles_in_scenario_config
            })
        
        for run_case_config in vehicle_run_cases:
            current_vehicle_case_folder_name = run_case_config["case_folder_suffix"]
            current_vehicles_to_load_list = run_case_config["vehicles_to_process"]
            
            active_scenario_name = scenario_name
            active_ego_vehicle_id = ego_id_for_filter
            active_vehicle_ids_list = current_vehicles_to_load_list
            active_timestamps_list_str = current_sequence_timestamps_str
            
            active_num_vehicles = len(active_vehicle_ids_list)
            active_num_timestamps_in_sequence = num_frames_for_sequence

            print(f"\n    [RUN_CASE]: {current_vehicle_case_folder_name} for timestamp_group {timestamp_group_key}")
            print(f"      Vehicles: {active_vehicle_ids_list}, Timestamps: {active_timestamps_list_str}")

            mode_path = os.path.join(BASE_PATH_GLOBAL, MODE_GLOBAL)
            scenario_fs_path = os.path.join(mode_path, active_scenario_name)

            image_names = []
            gt_extrinsics_ordered = []
            camera_names_ordered = []
            yaml_data_cache = {}

            for timestamp_val_str in active_timestamps_list_str:
                for vehicle_id_val in active_vehicle_ids_list:
                    yaml_key = (vehicle_id_val, timestamp_val_str)
                    if yaml_key not in yaml_data_cache:
                        yaml_file_path = os.path.join(scenario_fs_path, vehicle_id_val, f"{timestamp_val_str}.yaml")
                        with open(yaml_file_path, 'r') as f:
                            yaml_content = yaml.load(f, Loader=yaml.FullLoader)
                        yaml_data_cache[yaml_key] = yaml_content
                    
                    current_vehicle_yaml_data = yaml_data_cache[yaml_key]
                    vehicle_fs_path = os.path.join(scenario_fs_path, vehicle_id_val)

                    for camera_name_val in CAMERAS_GLOBAL:
                        if CAMERA_ENABLED_GLOBAL.get(camera_name_val, False):
                            image_file_path = os.path.join(vehicle_fs_path, f"{timestamp_val_str}_{camera_name_val}.png")
                            image_names.append(image_file_path)
                            camera_names_ordered.append(f"{vehicle_id_val}_{timestamp_val_str}_{camera_name_val}")

                            final_gt_W_C = None 
                            if current_vehicle_yaml_data and camera_name_val in current_vehicle_yaml_data:
                                if 'extrinsic' in current_vehicle_yaml_data[camera_name_val]:
                                    M_yaml_extrinsic_L_C = np.array(current_vehicle_yaml_data[camera_name_val]['extrinsic'])
                                    final_gt_W_C = M_yaml_extrinsic_L_C
                            gt_extrinsics_ordered.append(final_gt_W_C)
            
            if not image_names:
                print(f"        No images collected for this run. Skipping.")
                continue

            print(f"        Processing {len(image_names)} images for {active_num_vehicles} vehicle(s) (Ego for TUM: {active_ego_vehicle_id}) and {active_num_timestamps_in_sequence} timestamps...")
            
            images = load_and_preprocess_images(image_names).to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    images_batch = images[None]
                    aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    pred_extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            gt_tum_lines = []
            pred_tum_lines = []
            total_images_output_from_model = pred_extrinsic.shape[1]

            for i in range(total_images_output_from_model):
                gt_matrix_4x4_W_C = gt_extrinsics_ordered[i]
                pred_matrix_3x4_tensor_W_C = pred_extrinsic[0, i]
                full_camera_name = camera_names_ordered[i]
                
                vehicle_id_from_name_parts = full_camera_name.split('_')
                vehicle_id_from_name = vehicle_id_from_name_parts[0]

                if gt_matrix_4x4_W_C is not None and vehicle_id_from_name == active_ego_vehicle_id:
                    pred_matrix_4x4_W_C = np.eye(4)
                    pred_matrix_4x4_W_C[:3, :] = pred_matrix_3x4_tensor_W_C.detach().cpu().numpy()
                    
                    tum_line_timestamp_id = i 

                    gt_line = matrix_to_tum_line(gt_matrix_4x4_W_C, tum_line_timestamp_id)
                    pred_line = matrix_to_tum_line(pred_matrix_4x4_W_C, tum_line_timestamp_id)

                    gt_tum_lines.append(gt_line)
                    pred_tum_lines.append(pred_line)
            
            output_dir_l1_scenario = os.path.join(BASE_OUTPUT_DIR_GLOBAL, active_scenario_name)
            output_dir_l2_ts_group = os.path.join(output_dir_l1_scenario, str(timestamp_group_key))
            final_output_directory_for_run = os.path.join(output_dir_l2_ts_group, current_vehicle_case_folder_name)

            os.makedirs(final_output_directory_for_run, exist_ok=True)

            gt_filename_path = os.path.join(final_output_directory_for_run, "gt_poses.txt")
            pred_filename_path = os.path.join(final_output_directory_for_run, "pred_poses.txt")

            with open(gt_filename_path, 'w') as f:
                for line in gt_tum_lines:
                    f.write(line + '\n')
            print(f"        Ground truth poses (Ego: {active_ego_vehicle_id}) saved to: {gt_filename_path} ({len(gt_tum_lines)} poses)")

            with open(pred_filename_path, 'w') as f:
                for line in pred_tum_lines:
                    f.write(line + '\n')
            print(f"        Predicted poses (Ego: {active_ego_vehicle_id}) saved to: {pred_filename_path} ({len(pred_tum_lines)} poses)")

    print("-----------------------------------")

print("\n--- All scenarios and cases processed ---")