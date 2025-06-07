import torch
import os
import yaml 
import numpy as np
import json # For saving aggregated results
# from scipy.spatial.transform import Rotation as R # Not needed if mat_to_quat is from vggt

# Assuming vggt utilities are correctly importable
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3 # For CO3D eval functions
from vggt.utils.rotation import mat_to_quat # For CO3D eval functions

# --- SCENARIOS Configuration ---
from scenarios_config import SCENARIOS 

# --- START OF CO3D EVALUATION HELPER FUNCTIONS ---
# These functions are copied or adapted from the CO3D evaluation script you provided.

def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.

    Args:
        pred_se3: Predicted SE(3) transformations
        gt_se3: Ground truth SE(3) transformations
        num_frames: Number of frames

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    # Compute relative camera poses between pairs
    # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(
        gt_se3[pair_idx_i2]
    )
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(
        pred_se3[pair_idx_i2]
    )

    # Compute the difference in rotation and translation
    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg


def align_to_first_camera(camera_poses):
    """
    Align all camera poses to the first camera's coordinate frame.

    Args:
        camera_poses: Tensor of shape (N, 4, 4) containing camera poses as SE3 transformations

    Returns:
        Tensor of shape (N, 4, 4) containing aligned camera poses
    """
    first_cam_extrinsic_inv = closed_form_inverse_se3(camera_poses[0][None])
    aligned_poses = torch.matmul(camera_poses, first_cam_extrinsic_inv)
    return aligned_poses

# --- Global Settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_inference_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
pose_math_dtype = torch.float64 

print("Loading model...")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval() 
print("Model loaded.")

BASE_PATH_GLOBAL = "dataset/v2x_vit" 
MODE_GLOBAL = "train"
CAMERAS_GLOBAL = ["camera0", "camera1", "camera2", "camera3"]
CAMERA_ENABLED_GLOBAL = {
    "camera0": True, "camera1": False, "camera2": False, "camera3": True
}
OUTPUT_JSON_FILE = "aggregated_vggt_official_metrics.json"

# --- Main Processing Loop ---
all_vggt_official_metrics_aggregator = {}

for scenario_name, scenario_config in SCENARIOS.items():
    print(f"\n[SCENARIO]: {scenario_name}")
    print("-----------------------------------")

    ego_id_for_yaml_filter = scenario_config["ego_vehicle_id"]
    all_vehicles_in_scenario_config = scenario_config["vehicle_ids"]
    scenario_fs_path_for_scenario = os.path.join(BASE_PATH_GLOBAL, MODE_GLOBAL, scenario_name)

    for timestamp_group_key, integer_timestamp_list in scenario_config["timestamps_data"].items():
        current_sequence_timestamps_str = [str(ts_int).zfill(6) for ts_int in integer_timestamp_list]
        # num_frames_in_original_group = len(current_sequence_timestamps_str) # This is the N for this specific sequence input
        
        # The actual number of views/frames for evaluation will depend on how many images
        # are collected AND have valid GTs.
        
        print(f"  [TIMESTAMP_GROUP_ID]: {timestamp_group_key} (Input Timestamps: {current_sequence_timestamps_str})")

        # YAML Pre-check (same as in poses.py)
        current_timestamp_group_data_is_valid = True
        pre_loaded_yaml_cache_for_group = {} 
        unique_vehicle_ids_for_group_check = set([ego_id_for_yaml_filter])
        if set(all_vehicles_in_scenario_config) != set([ego_id_for_yaml_filter]):
            unique_vehicle_ids_for_group_check.update(all_vehicles_in_scenario_config)

        for ts_str_check in current_sequence_timestamps_str:
            if not current_timestamp_group_data_is_valid: break 
            for veh_id_check in unique_vehicle_ids_for_group_check:
                if not current_timestamp_group_data_is_valid: break 
                yaml_file_path_to_check = os.path.join(scenario_fs_path_for_scenario, veh_id_check, f"{ts_str_check}.yaml")
                yaml_dict_key_for_cache = (veh_id_check, ts_str_check)
                if yaml_dict_key_for_cache not in pre_loaded_yaml_cache_for_group:
                    try:
                        with open(yaml_file_path_to_check, 'r', encoding='utf-8') as f_check:
                            yaml_content_loaded = yaml.safe_load(f_check)
                        pre_loaded_yaml_cache_for_group[yaml_dict_key_for_cache] = yaml_content_loaded
                    except Exception as e:
                        print(f"      PRE-CHECK ERROR: Failed for YAML {yaml_file_path_to_check}: {e}")
                        current_timestamp_group_data_is_valid = False; break
            if not current_timestamp_group_data_is_valid: break
        
        if not current_timestamp_group_data_is_valid:
            print(f"    Skipping entire timestamp_group_key '{timestamp_group_key}' for scenario '{scenario_name}' due to YAML pre-check failures.")
            continue

        # Vehicle run cases
        vehicle_run_cases = []
        vehicle_run_cases.append({"case_name_suffix": "1_vehicle", "vehicles_to_process": [ego_id_for_yaml_filter]})
        if set(all_vehicles_in_scenario_config) != set([ego_id_for_yaml_filter]):
            vehicle_run_cases.append({
                "case_name_suffix": f"{len(all_vehicles_in_scenario_config)}_vehicles",
                "vehicles_to_process": all_vehicles_in_scenario_config
            })
        
        for run_case_config in vehicle_run_cases:
            current_vehicle_case_name = run_case_config["case_name_suffix"]
            current_vehicles_to_load = run_case_config["vehicles_to_process"]
            
            print(f"\n    [RUN_CASE]: {current_vehicle_case_name} for timestamp_group {timestamp_group_key}")

            run_case_image_names = []
            run_case_gt_matrices_np = [] # List of 4x4 T_W_C GT numpy matrices
            
            # Collect all image views and their corresponding GTs for this run case and sequence
            for ts_str in current_sequence_timestamps_str:
                for veh_id in current_vehicles_to_load: # Vehicles providing input images
                    yaml_content = pre_loaded_yaml_cache_for_group.get((veh_id, ts_str))
                    if yaml_content is None: continue # Should be caught by pre-check

                    vehicle_image_folder = os.path.join(scenario_fs_path_for_scenario, veh_id)
                    for cam_name in CAMERAS_GLOBAL:
                        if CAMERA_ENABLED_GLOBAL.get(cam_name, False):
                            img_path = os.path.join(vehicle_image_folder, f"{ts_str}_{cam_name}.png")
                            if not os.path.exists(img_path): continue

                            gt_matrix_4x4 = None
                            # GT extrinsics are from the YAML of the vehicle *owning* the camera
                            if cam_name in yaml_content and 'extrinsic' in yaml_content[cam_name]:
                                try:
                                    ext_data = yaml_content[cam_name]['extrinsic']
                                    gt_m_np = np.array(ext_data, dtype=np.float64) # Ensure consistent dtype
                                    if gt_m_np.shape == (4, 4):
                                        gt_matrix_4x4 = gt_m_np
                                    elif gt_m_np.shape == (3, 4):
                                        gt_matrix_4x4 = np.eye(4, dtype=np.float64)
                                        gt_matrix_4x4[:3, :] = gt_m_np
                                except Exception: pass # gt_matrix_4x4 remains None
                            
                            if gt_matrix_4x4 is not None: # Only if GT is valid
                                run_case_image_names.append(img_path)
                                run_case_gt_matrices_np.append(gt_matrix_4x4)
            
            num_views_for_eval = len(run_case_image_names)
            if num_views_for_eval < 2: # Relative pose error needs at least 2 views
                print(f"        Skipping {current_vehicle_case_name}: Needs at least 2 views with valid GT, found {num_views_for_eval}.")
                continue
            
            print(f"        Processing {num_views_for_eval} views with valid GT for CO3D-style evaluation...")

            # Model Inference
            images = load_and_preprocess_images(run_case_image_names).to(device)
            pred_matrices_list_np = [] # List of 4x4 T_W_C Predicted numpy matrices
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=model_inference_dtype):
                    images_batch = images[None]
                    aggregated_tokens_list, _ = model.aggregator(images_batch)
                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    pred_extrinsic_3x4_tensor_b, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            
            if pred_extrinsic_3x4_tensor_b.shape[1] == num_views_for_eval:
                for k_idx in range(num_views_for_eval):
                    pred_m_3x4_torch = pred_extrinsic_3x4_tensor_b[0, k_idx]
                    pred_m_4x4_np = np.eye(4, dtype=np.float64) # Match dtype with GT
                    pred_m_4x4_np[:3, :] = pred_m_3x4_torch.cpu().to(torch.float64).numpy() # Convert to float64 for consistency
                    pred_matrices_list_np.append(pred_m_4x4_np)
            else:
                print(f"        Error: Mismatch in pred tensor views ({pred_extrinsic_3x4_tensor_b.shape[1]}) and images with GT ({num_views_for_eval}). Skipping eval.")
                continue
            
            # Perform CO3D-style Evaluation
            gt_se3_tensor_eval = torch.tensor(np.stack(run_case_gt_matrices_np), dtype=pose_math_dtype, device=device)
            pred_se3_tensor_eval = torch.tensor(np.stack(pred_matrices_list_np), dtype=pose_math_dtype, device=device)

            aligned_gt_se3 = align_to_first_camera(gt_se3_tensor_eval)
            aligned_pred_se3 = align_to_first_camera(pred_se3_tensor_eval)

            rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                aligned_pred_se3, aligned_gt_se3, num_views_for_eval
            )
            
            if rel_rangle_deg.numel() > 0 and rel_tangle_deg.numel() > 0:
                metrics = {}
                # Ensure errors are on CPU and are numpy arrays for calculate_auc_np
                r_error_np_cpu = rel_rangle_deg.detach().cpu().numpy()
                t_error_np_cpu = rel_tangle_deg.detach().cpu().numpy()

                for N_deg_thresh in [1, 3, 5, 10, 15, 30]:
                    metrics[f"Racc@{N_deg_thresh}deg"] = (rel_rangle_deg < N_deg_thresh).float().mean().item()
                    metrics[f"Tacc@{N_deg_thresh}deg"] = (rel_tangle_deg < N_deg_thresh).float().mean().item()
                    auc_val, _ = calculate_auc_np(r_error_np_cpu, t_error_np_cpu, max_threshold=N_deg_thresh)
                    metrics[f"AUC@{N_deg_thresh}deg"] = auc_val
                
                metrics["num_evaluated_view_pairs_for_rpe"] = rel_rangle_deg.numel() # Number of pairs used for RPE
                metrics["num_input_views_in_sequence"] = num_views_for_eval
                print(f"        Metrics for {current_vehicle_case_name}: Racc@5={metrics['Racc@5deg']:.4f}, Tacc@5={metrics['Tacc@5deg']:.4f}, AUC@30={metrics['AUC@30deg']:.4f}")

                # Aggregate results
                s_data = all_vggt_official_metrics_aggregator.setdefault(scenario_name, {
                    "vehicle_ids": scenario_config["vehicle_ids"],
                    "ego_vehicle_id": ego_id_for_yaml_filter,
                    "base_timestamp_ref": scenario_config.get("base_timestamp_ref"),
                    "end_timestamp_ref": scenario_config.get("end_timestamp_ref"),
                    "sequence_group_results": {}
                })
                g_data = s_data["sequence_group_results"].setdefault(str(timestamp_group_key), {
                    "timestamps_in_sequence_str": current_sequence_timestamps_str,
                    "vggt_official_eval_results": {}
                })
                g_data["vggt_official_eval_results"][current_vehicle_case_name] = metrics
            else:
                print(f"        No relative errors computed for {current_vehicle_case_name} (num_input_views={num_views_for_eval}). Cannot calculate metrics.")
    print("-----------------------------------")

if all_vggt_official_metrics_aggregator:
    with open(OUTPUT_JSON_FILE, 'w') as f_json:
        json.dump(all_vggt_official_metrics_aggregator, f_json, indent=4)
    print(f"\nVGGT Official evaluation metrics saved to {OUTPUT_JSON_FILE}")
else:
    print("\nNo VGGT Official evaluation metrics were aggregated.")

print("\n--- VGGT Official Evaluation Script Finished ---")