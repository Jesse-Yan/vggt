import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil

# Assuming vggt utilities are correctly importable
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

# --- Helper Functions (read_file_list_for_associate, associate_timestamps, matrix_to_tum_line, 
# --- get_rgb_data_for_association, parse_tum_groundtruth_for_association,
# --- _collect_image_paths_for_inference, _run_vggt_inference_internal,
# --- _generate_and_save_tum_files_internal) remain the same as in the previous complete code block.
# --- For brevity, I will omit them here but they should be included in your final script.

# --- [Re-paste helper functions from the previous complete code block here if running standalone] ---
# --- Helper Function from associate.py logic (slightly adapted) ---
def read_file_list_for_associate(filename):
    try:
        with open(filename, 'r') as file:
            data = file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return {}
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return {}
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    processed_list = []
    for line in lines:
        if len(line) > 0 and line[0] != "#":
            parts = [v.strip() for v in line.split(" ") if v.strip() != ""]
            if len(parts) > 1:
                try:
                    timestamp = float(parts[0])
                    data_values = parts[1:]
                    processed_list.append((timestamp, data_values))
                except ValueError:
                    print(f"Warning: Could not parse line in {filename}: {line}")
    return dict(processed_list)

def associate_timestamps(first_data_dict, second_data_dict, offset, max_difference):
    first_keys = set(first_data_dict.keys())
    second_keys = set(second_data_dict.keys())
    potential_matches = []
    for a in first_keys:
        for b in second_keys:
            diff = abs(a - (b + offset))
            if diff < max_difference:
                potential_matches.append((diff, a, b))
    potential_matches.sort()
    
    first_keys_available = set(first_data_dict.keys())
    second_keys_available = set(second_data_dict.keys())
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys_available and b in second_keys_available:
            matches.append((a, b))
            first_keys_available.remove(a)
            second_keys_available.remove(b)
    matches.sort()
    return matches

def matrix_to_tum_line(matrix_4x4, timestamp_str):
    t = matrix_4x4[:3, 3]
    R_mat = matrix_4x4[:3, :3]
    q = R.from_matrix(R_mat).as_quat()
    return f"{timestamp_str} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}"

def get_rgb_data_for_association(rgb_folder_path):
    if not os.path.isdir(rgb_folder_path):
        print(f"Error: RGB folder not found at {rgb_folder_path}")
        return {}
    image_files = [f for f in os.listdir(rgb_folder_path) if f.endswith(".png")]
    rgb_data_assoc = {}
    for fname in image_files:
        try:
            timestamp_str = os.path.splitext(fname)[0]
            timestamp_float = float(timestamp_str)
            relative_image_path = os.path.join(os.path.basename(rgb_folder_path), fname)
            rgb_data_assoc[timestamp_float] = [relative_image_path] 
        except ValueError:
            print(f"Warning: Could not parse timestamp from RGB filename {fname}")
    return rgb_data_assoc

def parse_tum_groundtruth_for_association(filepath):
    data_for_associate = {} 
    float_to_original_stamp_str = {} 
    if not os.path.exists(filepath):
        print(f"Warning: Ground truth file not found at {filepath}")
        return data_for_associate, float_to_original_stamp_str
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) == 8:
                original_stamp_str = parts[0]
                pose_values_str_list = parts[1:]
                try:
                    ts_float = float(original_stamp_str)
                    data_for_associate[ts_float] = pose_values_str_list
                    float_to_original_stamp_str[ts_float] = original_stamp_str
                except ValueError: print(f"Warning: Could not convert GT timestamp {original_stamp_str} to float in {filepath}")
            else: print(f"Warning: Malformed line in TUM file {filepath}: {line}")
    return data_for_associate, float_to_original_stamp_str

def _collect_image_paths_for_inference(matched_pairs_subset, rgb_data_assoc_map, dataset_root_path_val):
    image_paths = []
    valid_matches_for_images = []
    for rgb_ts_f, gt_ts_f in matched_pairs_subset:
        relative_image_path = rgb_data_assoc_map.get(rgb_ts_f, [None])[0]
        if relative_image_path:
            full_image_path = os.path.join(dataset_root_path_val, relative_image_path)
            if os.path.exists(full_image_path):
                image_paths.append(full_image_path)
                valid_matches_for_images.append((rgb_ts_f, gt_ts_f))
            else:
                print(f"  Warning: Image file not found for matched RGB timestamp {rgb_ts_f}: {full_image_path}")
        else:
            print(f"  Warning: No image path found in rgb_data_assoc_map for RGB timestamp {rgb_ts_f}")
    return image_paths, valid_matches_for_images

def _run_vggt_inference_internal(image_paths_list, model_obj, device_obj, dtype_obj):
    if not image_paths_list:
        return None
    print(f"    Loading {len(image_paths_list)} images for VGGT inference...")
    images = load_and_preprocess_images(image_paths_list).to(device_obj)
    print("    Running VGGT model inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype_obj):
            images_batch = images[None]
            aggregated_tokens_list, _ = model_obj.aggregator(images_batch)
            pose_enc = model_obj.camera_head(aggregated_tokens_list)[-1]
            pred_extrinsic_tensors, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    print("    Model inference complete.")
    return pred_extrinsic_tensors

def _generate_and_save_tum_files_internal(
    output_dir,
    matched_pairs_for_output, 
    predictions_tensor, 
    all_gt_data_assoc, 
    all_gt_float_to_original_stamp_str,
    run_description 
    ):
    final_gt_tum_lines = []
    final_pred_tum_lines = []

    if predictions_tensor is None or predictions_tensor.shape[1] != len(matched_pairs_for_output):
        print(f"    Error: Mismatch or missing predictions for {run_description}. Predicted: {predictions_tensor.shape[1] if predictions_tensor is not None else 'None'}, Expected for output: {len(matched_pairs_for_output)}")
        return

    for i in range(len(matched_pairs_for_output)):
        _rgb_ts_f, gt_ts_f_current = matched_pairs_for_output[i]
        output_timestamp_str = all_gt_float_to_original_stamp_str.get(gt_ts_f_current)
        gt_pose_values_str_list = all_gt_data_assoc.get(gt_ts_f_current)

        if output_timestamp_str and gt_pose_values_str_list:
            gt_line = f"{output_timestamp_str} {' '.join(gt_pose_values_str_list)}"
            final_gt_tum_lines.append(gt_line)
            pred_matrix_3x4 = predictions_tensor[0, i]
            pred_matrix_4x4 = np.eye(4)
            pred_matrix_4x4[:3, :] = pred_matrix_3x4.detach().cpu().numpy()
            pred_line = matrix_to_tum_line(pred_matrix_4x4, output_timestamp_str)
            final_pred_tum_lines.append(pred_line)
        else:
            print(f"    Warning: Missing GT data or original timestamp string for GT float ts {gt_ts_f_current} during output generation for {run_description}.")

    os.makedirs(output_dir, exist_ok=True)
    gt_output_filepath = os.path.join(output_dir, "gt_poses.txt")
    pred_output_filepath = os.path.join(output_dir, "pred_poses.txt")
    with open(gt_output_filepath, 'w') as f: f.writelines(line + '\n' for line in final_gt_tum_lines)
    print(f"    Ground truth poses for {run_description} saved to: {gt_output_filepath} ({len(final_gt_tum_lines)} lines)")
    with open(pred_output_filepath, 'w') as f: f.writelines(line + '\n' for line in final_pred_tum_lines)
    print(f"    Predicted poses for {run_description} saved to: {pred_output_filepath} ({len(final_pred_tum_lines)} lines)")

# --- Main Processing Function with Strides (parameter name changed) ---
def process_dataset_with_strides(
    dataset_root_path, 
    max_frames, ego_stride, context_stride, # Changed 'inference_stride' to 'context_stride'
    base_output_directory,
    loaded_model, torch_device, torch_dtype,
    association_max_difference=0.02
    ):
    dataset_name = os.path.basename(os.path.normpath(dataset_root_path))
    # Updated print statement to use new parameter name
    print(f"\n[DATASET PROCESSING]: {dataset_name} with Strides (max_frames={max_frames}, ego_s={ego_stride}, context_s={context_stride})")

    rgb_folder = os.path.join(dataset_root_path, "rgb")
    gt_filepath = os.path.join(dataset_root_path, "groundtruth.txt")

    rgb_data_assoc = get_rgb_data_for_association(rgb_folder)
    gt_data_for_assoc, gt_float_to_original_stamp_str = parse_tum_groundtruth_for_association(gt_filepath)

    if not rgb_data_assoc or not gt_data_for_assoc:
        print(f"  Skipping {dataset_name}: Missing RGB or GT data for association.")
        return

    print(f"  Associating RGB and GT timestamps for {dataset_name}...")
    matched_timestamp_pairs = associate_timestamps(rgb_data_assoc, gt_data_for_assoc, 0.0, association_max_difference)
    
    if not matched_timestamp_pairs:
        print(f"  Skipping {dataset_name}: No timestamp pairs matched.")
        return
    print(f"  Found {len(matched_timestamp_pairs)} matched_timestamp_pairs.")

    candidate_matches = matched_timestamp_pairs[:min(max_frames, len(matched_timestamp_pairs))]
    if not candidate_matches:
        print(f"  Skipping {dataset_name}: No candidate matches after applying max_frames ({max_frames}).")
        return
    print(f"  Selected {len(candidate_matches)} candidate matches (up to max_frames={max_frames}).")

    # --- Run 1: Ego Stride Run ---
    print("\n  --- Running: Ego Stride Based ---")
    ego_stride_input_matches_raw = candidate_matches[::ego_stride]
    
    if not ego_stride_input_matches_raw:
        print("    No frames selected for Ego Stride Run. Skipping.")
    else:
        image_paths_ego, ego_stride_input_matches = _collect_image_paths_for_inference(
            ego_stride_input_matches_raw, rgb_data_assoc, dataset_root_path
        )
        if image_paths_ego:
            pred_extrinsic_ego_run = _run_vggt_inference_internal(image_paths_ego, loaded_model, torch_device, torch_dtype)
            if pred_extrinsic_ego_run is not None:
                # Output directory reflects parameters used for this run's input
                output_dir_ego = os.path.join(base_output_directory, dataset_name, f"max{max_frames}_input_ego_stride{ego_stride}")
                _generate_and_save_tum_files_internal(
                    output_dir_ego, ego_stride_input_matches, pred_extrinsic_ego_run,
                    gt_data_for_assoc, gt_float_to_original_stamp_str,
                    f"EgoStrideRunInput (max{max_frames},es{ego_stride})" # Description for output
                )
        else:
            print("    No valid images for Ego Stride Run after path validation.")


    # --- Run 2: Context Stride Run (Output filtered by Ego Stride) ---
    # Using 'context_stride' for variable names and print statements
    print("\n  --- Running: Context Stride Based (output filtered by Ego Stride) ---")
    context_stride_input_matches_raw = candidate_matches[::context_stride]

    if not context_stride_input_matches_raw:
        print("    No frames selected for Context Stride Run input. Skipping.")
    else:
        image_paths_context, context_stride_input_matches = _collect_image_paths_for_inference(
            context_stride_input_matches_raw, rgb_data_assoc, dataset_root_path
        ) 
        
        if image_paths_context:
            pred_extrinsic_context_run = _run_vggt_inference_internal(image_paths_context, loaded_model, torch_device, torch_dtype)
            
            if pred_extrinsic_context_run is not None:
                output_matches_for_context_run_filtered = []
                predictions_for_output_list_filtered = [] 

                candidate_matches_with_indices = {match_pair: idx for idx, match_pair in enumerate(candidate_matches)}

                for j, current_context_match_pair in enumerate(context_stride_input_matches):
                    original_candidate_idx = candidate_matches_with_indices.get(current_context_match_pair)
                    
                    if original_candidate_idx is not None and original_candidate_idx % ego_stride == 0:
                        output_matches_for_context_run_filtered.append(current_context_match_pair)
                        predictions_for_output_list_filtered.append(pred_extrinsic_context_run[0, j]) 

                if output_matches_for_context_run_filtered:
                    pred_tensors_for_output_final_filtered = torch.stack(predictions_for_output_list_filtered, dim=0) 
                    if pred_tensors_for_output_final_filtered.ndim == 3 : 
                         pred_tensors_for_output_final_filtered = pred_tensors_for_output_final_filtered.unsqueeze(0)

                    # Output directory reflects parameters used for this run's input and output filtering
                    output_dir_context = os.path.join(base_output_directory, dataset_name, f"max{max_frames}_input_context_stride{context_stride}_out_ego_stride{ego_stride}")
                    _generate_and_save_tum_files_internal(
                        output_dir_context, output_matches_for_context_run_filtered, pred_tensors_for_output_final_filtered,
                        gt_data_for_assoc, gt_float_to_original_stamp_str,
                        f"ContextStrideRunFilteredOutput (max{max_frames},cs{context_stride},es{ego_stride})"
                    )
                else:
                    print("    No frames from Context Stride Run matched the Ego Stride criteria for output.")
        else:
            print("    No valid images for Context Stride Run after path validation.")


if __name__ == "__main__":
    # --- Configuration
    DATASET_ROOT_PATH = "dataset/rgbd_dataset_freiburg1_360" 
    MAX_FRAMES_TO_PROCESS = 100
    EGO_STRIDE = 2               
    CONTEXT_STRIDE = 1
    
    BASE_OUTPUT_DIRECTORY = "evo_tum" 
    ASSOCIATION_MAX_DIFFERENCE_SEC = 0.02
    # --- End of Configuration ---

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("Loading VGGT model...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    print("Model loaded.")
    
    process_dataset_with_strides(
        dataset_root_path=DATASET_ROOT_PATH,
        max_frames=MAX_FRAMES_TO_PROCESS,
        ego_stride=EGO_STRIDE,
        context_stride=CONTEXT_STRIDE,
        base_output_directory=BASE_OUTPUT_DIRECTORY,
        loaded_model=vggt_model,
        torch_device=device,
        torch_dtype=dtype,
        association_max_difference=ASSOCIATION_MAX_DIFFERENCE_SEC
    )

    print("\n--- Script finished ---")