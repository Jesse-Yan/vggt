import subprocess
import os
import zipfile
import json
import shutil

# SCENARIOS dictionary (as defined and confirmed previously with integer timestamps)
from scenarios_config import SCENARIOS  # Import the SCENARIOS dict from your config file

TEMP_EVO_OUTPUT_DIR_NAME = "evo_tmp" # Defined earlier

def find_evo_file_pairs_and_context(base_dir, scenarios_definition):
    """
    Walks through the base_dir to find gt_poses.txt and pred_poses.txt pairs
    and collects context information including the processed timestamp list.
    """
    file_pairs_with_context = []
    if not os.path.isdir(base_dir):
        print(f"Warning: Base directory '{base_dir}' does not exist. No files to process.")
        return file_pairs_with_context

    for scenario_name in os.listdir(base_dir):
        scenario_path = os.path.join(base_dir, scenario_name)
        if not os.path.isdir(scenario_path):
            continue
        
        # Check if this scenario_name exists in our SCENARIOS definition
        if scenario_name not in scenarios_definition:
            # print(f"  Skipping directory '{scenario_name}' as it's not defined in SCENARIOS config.")
            continue

        for ts_group_id_str in os.listdir(scenario_path):
            ts_group_path = os.path.join(scenario_path, ts_group_id_str)
            if not os.path.isdir(ts_group_path):
                continue

            # Convert ts_group_id_str to the type of key used in SCENARIOS (int in our case)
            # This assumes keys in SCENARIOS[...]["timestamps_data"] are integers like 0, 1...
            # If keys could be strings like "groupA", this conversion needs to be more robust
            # or the key type needs to be consistent. For now, assuming int keys.
            try:
                # Try to convert folder name (ts_group_id_str) to int if SCENARIOS uses int keys
                # This is a potential mismatch point if folder names don't map to dict keys
                ts_group_id_key_type = None
                # Check type of keys in the reference SCENARIOS dict for this scenario's timestamps_data
                if scenario_name in scenarios_definition and \
                   "timestamps_data" in scenarios_definition[scenario_name] and \
                   scenarios_definition[scenario_name]["timestamps_data"]:
                    # Get a sample key to infer type
                    sample_key = next(iter(scenarios_definition[scenario_name]["timestamps_data"]))
                    if isinstance(sample_key, int):
                        ts_group_id_key_type = int(ts_group_id_str)
                    else: # Assume string key if not int
                        ts_group_id_key_type = ts_group_id_str
                else: # Fallback or error if no timestamps_data defined for scenario
                    # print(f"  Warning: No timestamps_data in SCENARIOS for {scenario_name}. Cannot map ts_group_id '{ts_group_id_str}'.")
                    continue
            except ValueError:
                # print(f"  Warning: Could not convert ts_group_id folder '{ts_group_id_str}' to expected key type for SCENARIOS. Skipping.")
                continue


            # Retrieve the original integer timestamp list for this group from SCENARIOS
            integer_timestamp_list = scenarios_definition.get(scenario_name, {}).get("timestamps_data", {}).get(ts_group_id_key_type)
            
            if integer_timestamp_list is None:
                # print(f"  Warning: No timestamp data found in SCENARIOS for {scenario_name} / group {ts_group_id_key_type}. Skipping path.")
                continue
            
            processed_timestamps_str_list = [str(ts_int).zfill(6) for ts_int in integer_timestamp_list]

            for vehicle_case_folder in os.listdir(ts_group_path):
                case_path = os.path.join(ts_group_path, vehicle_case_folder)
                if not os.path.isdir(case_path):
                    continue

                gt_file = os.path.join(case_path, "gt_poses.txt")
                pred_file = os.path.join(case_path, "pred_poses.txt")

                if os.path.isfile(gt_file) and os.path.isfile(pred_file):
                    context = {
                        "scenario_name": scenario_name,
                        "timestamp_group_id": ts_group_id_key_type, # Store the key as used in dict
                        "vehicle_case": vehicle_case_folder,
                        "gt_file_path": gt_file,
                        "pred_file_path": pred_file,
                        "processed_timestamps_list": processed_timestamps_str_list
                    }
                    file_pairs_with_context.append(context)
                # else:
                    # print(f"  Warning: gt_poses.txt or pred_poses.txt not found in {case_path}")
    
    return file_pairs_with_context

def run_single_evo_ape_and_get_stats(gt_file_path, pred_file_path, temp_output_dir, context_info_str=""):
    temp_zip_filename = "ape_results_temp.zip"
    temp_zip_full_path = os.path.join(temp_output_dir, temp_zip_filename)

    command = [
        "evo_ape", "tum",
        gt_file_path,
        pred_file_path,
        "-va", "-s", "-a",
        "--save_results", temp_zip_full_path
    ]
    stats_dict = None
    printable_context = f" (Context: {context_info_str})" if context_info_str else ""

    try:
        print(f"    Running EVO APE for: {os.path.basename(gt_file_path)}, {os.path.basename(pred_file_path)}{printable_context}")
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        
        if os.path.exists(temp_zip_full_path):
            try:
                with zipfile.ZipFile(temp_zip_full_path, 'r') as zip_ref:
                    if 'stats.json' in zip_ref.namelist():
                        json_bytes = zip_ref.read('stats.json')
                        stats_dict = json.loads(json_bytes.decode('utf-8'))
                    else:
                        print(f"      Warning: 'stats.json' not found in {temp_zip_full_path}{printable_context}.")
            except zipfile.BadZipFile:
                print(f"      Error: Bad zip file created by EVO: {temp_zip_full_path}{printable_context}.")
            except json.JSONDecodeError:
                print(f"      Error: Could not decode 'stats.json' from {temp_zip_full_path}{printable_context}.")
            except Exception as e:
                print(f"      Error processing zip file {temp_zip_full_path}: {e}{printable_context}.")
            finally:
                if os.path.exists(temp_zip_full_path):
                    os.remove(temp_zip_full_path)
        else: 
            print(f"      Warning: EVO command seemed to succeed but zip was not created at {temp_zip_full_path}{printable_context}.")
            if result: 
                 print(f"      EVO stdout (first 500 chars): {result.stdout[:500]}") # Limit output
                 print(f"      EVO stderr (first 500 chars): {result.stderr[:500]}") # Limit output

    except subprocess.CalledProcessError as e:
        print(f"      Error running EVO APE for {os.path.basename(gt_file_path)} / {os.path.basename(pred_file_path)}{printable_context}.")
        # print(f"      Command failed: {' '.join(e.cmd)}") # Command can be long with paths
        print(f"      Return code: {e.returncode}")
        print(f"      Stdout (first 500 chars): {e.stdout[:500]}")
        print(f"      Stderr (first 500 chars): {e.stderr[:500]}")
        if os.path.exists(temp_zip_full_path):
            os.remove(temp_zip_full_path)
        return None 
    except FileNotFoundError: 
        print(f"      Error: 'evo_ape' command not found. Make sure EVO is installed and in your PATH.{printable_context}")
        return None
    except Exception as e: 
        print(f"      An unexpected error occurred for {os.path.basename(gt_file_path)}{printable_context}: {e}")
        if os.path.exists(temp_zip_full_path):
            os.remove(temp_zip_full_path)
        return None
    return stats_dict

# --- Main script execution flow ---
if __name__ == "__main__":
    # This should be the path to the directory structure created by the previous script
    # e.g., "evo_vggt" if your previous BASE_OUTPUT_DIR_GLOBAL was "evo_vggt"
    tum_files_base_directory = "evo_vggt" # Or your actual output directory

    temp_dir_main_path = TEMP_EVO_OUTPUT_DIR_NAME 
    os.makedirs(temp_dir_main_path, exist_ok=True)
    print(f"Temporary EVO output directory: {os.path.abspath(temp_dir_main_path)}")

    # 1. Find all file pairs and their context
    # Pass SCENARIOS dict to resolve timestamp lists
    all_file_pairs = find_evo_file_pairs_and_context(tum_files_base_directory, SCENARIOS)
    
    if not all_file_pairs:
        print(f"No file pairs found in '{tum_files_base_directory}'. Exiting.")
    else:
        print(f"\nFound {len(all_file_pairs)} file pairs to process for EVO APE.")

    # 2. Initialize the dictionary to store aggregated results
    aggregated_results_json = {}

    # 3. Process each file pair
    for file_info in all_file_pairs:
        s_name = file_info['scenario_name']
        ts_group_id = file_info['timestamp_group_id'] # This is now the key as in SCENARIOS (e.g., int 0)
        v_case = file_info['vehicle_case']
        gt_p = file_info['gt_file_path']
        pred_p = file_info['pred_file_path']
        seq_timestamps_str_list = file_info['processed_timestamps_list']

        context_log_str = f"Scenario: {s_name}, Group: {ts_group_id}, Case: {v_case}"
        
        ape_stats = run_single_evo_ape_and_get_stats(gt_p, pred_p, temp_dir_main_path, context_log_str)
        
        if ape_stats:
            # Ensure scenario level exists
            aggregated_results_json.setdefault(s_name, {
                "vehicle_ids": SCENARIOS[s_name]["vehicle_ids"], # Copy from SCENARIOS
                "ego_vehicle_id": SCENARIOS[s_name]["ego_vehicle_id"], # Copy
                "base_timestamp_ref": SCENARIOS[s_name]["base_timestamp_ref"], # Optional carry-over
                "end_timestamp_ref": SCENARIOS[s_name]["end_timestamp_ref"],   # Optional carry-over
                "sequence_group_results": {}
            })
            
            # Ensure timestamp_group_id level exists
            aggregated_results_json[s_name]["sequence_group_results"].setdefault(ts_group_id, {
                "timestamps_in_sequence": seq_timestamps_str_list, # Store the actual processed timestamps
                "evo_results": {}
            })
            
            # Store APE results for the specific vehicle case
            aggregated_results_json[s_name]["sequence_group_results"][ts_group_id]["evo_results"][v_case] = ape_stats
        else:
            print(f"      Skipping results for {context_log_str} due to previous errors.")

    # 4. Save the aggregated results to a JSON file
    output_json_filename = "all_aggregated_evo_ape_results.json"
    if aggregated_results_json:
        with open(output_json_filename, 'w') as f:
            json.dump(aggregated_results_json, f, indent=4)
        print(f"\nAggregated EVO APE results saved to: {output_json_filename}")
    else:
        print("\nNo results were aggregated to save.")

    # 5. Clean up the temporary directory
    try:
        if os.path.exists(temp_dir_main_path):
            shutil.rmtree(temp_dir_main_path)
            print(f"\nTemporary EVO output directory '{temp_dir_main_path}' removed.")
    except OSError as e:
        print(f"Error removing directory {temp_dir_main_path} : {e.strerror}")

    print("\n--- EVO APE Processing Script Finished ---")