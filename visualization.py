import yaml # PyYAML library needed: pip install pyyaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os, sys
from pathlib import Path # Using pathlib for easier path manipulation
from collections import defaultdict
import traceback

# --- Plotting Function ---
def plot_bev(timestamp, vehicles_to_plot, scenario_id, save_dir):
    """
    Generates and saves a Bird's-Eye View plot for a single timestamp.

    Args:
        timestamp (str): The timestamp string (e.g., '000068').
        vehicles_to_plot (dict): Dictionary mapping vehicle_id to its state dict
                                 {'x', 'y', 'yaw', 'length', 'width', 'role'}.
        scenario_id (str): The scenario identifier.
        save_dir (str): Directory to save the plot image.
    """
    if not vehicles_to_plot:
        print(f"Info: No vehicles found to plot for timestamp {timestamp}.")
        return

    df = pd.DataFrame.from_dict(vehicles_to_plot, orient='index')
    df['id'] = df.index # Add vehicle ID as a column for labeling

    fig, ax = plt.subplots(figsize=(12, 12))

    # Define colors based on roles ('provider' or 'background')
    color_map = {
        'provider': 'blue',    # Data providing vehicles are blue
        'background': 'gray',  # Other observed vehicles are gray
    }
    default_color = 'black' # Fallback

    all_x = df['x'].tolist()
    all_y = df['y'].tolist()

    # Plot Boxes and Labels
    for index, row in df.iterrows():
        x_center, y_center = row['x'], row['y']
        length, width = row['length'], row['width']
        yaw = row['yaw']
        role = row['role']
        current_id = row['id']

        half_length, half_width = length / 2, width / 2
        corners_local = np.array([[half_length, half_width], [half_length, -half_width], [-half_length, -half_width], [-half_length, half_width]])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners_world = corners_local @ rotation.T + np.array([x_center, y_center])

        face_color = color_map.get(role, default_color)
        polygon = patches.Polygon(corners_world, closed=True, edgecolor=face_color, facecolor=face_color, alpha=0.6)
        ax.add_patch(polygon)
        ax.text(x_center, y_center, str(current_id), ha='center', va='center', fontsize=6, color='black')

    # --- Final Touches ---
    if not all_x or not all_y: x_min, x_max, y_min, y_max = -50, 50, -50, 50
    else:
        padding = 20
        x_min, x_max = min(all_x) - padding, max(all_x) + padding
        y_min, y_max = min(all_y) - padding, max(all_y) + padding
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X coordinate (m)"); ax.set_ylabel("Y coordinate (m)")
    ax.set_title(f"V2X-ViT BEV - Scenario {scenario_id} - Timestamp {timestamp}")
    ax.set_aspect('equal', adjustable='box'); ax.grid(True)

    # Create legend
    legend_handles = [
        patches.Patch(color=color_map['provider'], label='Data Provider', alpha=0.6),
        patches.Patch(color=color_map['background'], label='Background Vehicle', alpha=0.6),
    ]
    ax.legend(handles=legend_handles, fontsize='small')

    # Save the plot
    if not os.path.exists(save_dir):
        try: os.makedirs(save_dir)
        except OSError as e: print(f"Error creating dir {save_dir}: {e}"); plt.close(fig); return
    save_path = os.path.join(save_dir, f"scene_{scenario_id}_ts_{timestamp}.png")
    try:
        plt.savefig(save_path, dpi=300)
    except Exception as e: print(f"Error saving plot for timestamp {timestamp}: {e}")
    plt.close(fig) # Close plot to free memory


# --- Main Processing Function ---
def process_scenario(scenario_id, base_data_dir="dataset/v2x_vit/train", save_dir_base="bev_plots_v2xvit"):
    """
    Processes all timestamps for a given scenario, generating BEV plots.
    """
    scenario_path = Path(base_data_dir) / scenario_id
    save_dir = Path(save_dir_base) / scenario_id

    if not scenario_path.is_dir():
        print(f"Error: Scenario directory not found: {scenario_path}")
        return

    print(f"Processing scenario: {scenario_id}")
    print(f"Plots will be saved in: {save_dir}")

    data_provider_ids = set()
    all_timestamps = set()
    vehicle_yaml_files = defaultdict(list) # timestamp -> list of yaml file paths

    print("Scanning for vehicles and timestamps...")
    # Find all vehicle IDs (subdirectories) and collect all yaml timestamps
    for vehicle_dir in scenario_path.iterdir():
        if vehicle_dir.is_dir():
            try:
                 provider_id = int(vehicle_dir.name) # Assuming folder name is integer ID
                 data_provider_ids.add(provider_id)
                 for yaml_file in vehicle_dir.glob('*.yaml'):
                     timestamp = yaml_file.stem # Filename without extension
                     all_timestamps.add(timestamp)
                     vehicle_yaml_files[timestamp].append(yaml_file)
            except ValueError:
                 print(f"Warning: Skipping non-integer directory name: {vehicle_dir.name}")
                 continue # Skip if folder name isn't a vehicle ID

    if not data_provider_ids:
        print("Error: No vehicle directories found in scenario path.")
        return
    if not all_timestamps:
        print("Error: No .yaml files found in any vehicle directory.")
        return

    sorted_timestamps = sorted(list(all_timestamps))
    print(f"Found {len(data_provider_ids)} data providers: {sorted(list(data_provider_ids))}")
    print(f"Found {len(sorted_timestamps)} unique timestamps.")

    # Process each timestamp
    for timestamp in sorted_timestamps:
        print(f"  Processing timestamp: {timestamp}")
        vehicles_to_plot = {} # Reset for each frame: vehicle_id -> state dict

        # Load data from all provider files for this timestamp
        for yaml_file_path in vehicle_yaml_files.get(timestamp, []):
            try:
                with open(yaml_file_path, 'r') as f:
                    observer_data = yaml.safe_load(f) # Load YAML content

                # Process vehicles observed by this observer
                observed_vehicles_dict = observer_data.get('vehicles', {})
                if isinstance(observed_vehicles_dict, dict):
                     for observed_id_str, observed_data in observed_vehicles_dict.items():
                         try:
                             observed_id = int(observed_id_str) # Convert key to int
                         except ValueError:
                             continue # Skip if key isn't a valid int ID

                         # Add to plot list only if not already processed for this frame
                         if observed_id not in vehicles_to_plot:
                             # Extract state (assuming keys exist based on user guarantee)
                             try:
                                 loc = observed_data['location']
                                 angle = observed_data['angle']
                                 extent = observed_data['extent']

                                 x, y = loc[0], loc[1]
                                 # Assuming index 1 is yaw and unit is degrees
                                 yaw = angle[1] * np.pi / 180
                                 # Assuming extent is half-size
                                 length = extent[0] * 2
                                 width = extent[1] * 2

                                 # Determine role
                                 role = 'provider' if observed_id in data_provider_ids else 'background'

                                 vehicles_to_plot[observed_id] = {
                                     'x': x, 'y': y, 'yaw': yaw,
                                     'length': length, 'width': width, 'role': role
                                 }
                             except (KeyError, IndexError, TypeError) as e:
                                 # Reduced error checks as requested, but log if something unexpected happens
                                 # print(f"Warn: Error extracting data for observed vehicle {observed_id} in {yaml_file_path}: {e}")
                                 pass # Silently skip malformed observed vehicle data

            except yaml.YAMLError as e:
                print(f"Error parsing YAML file {yaml_file_path}: {e}")
            except IOError as e:
                 print(f"Error reading file {yaml_file_path}: {e}")
            except Exception as e:
                 print(f"Unexpected error processing file {yaml_file_path}: {e}")
                 traceback.print_exc()


        # Plot the aggregated data for this timestamp
        plot_bev(
            timestamp=timestamp,
            vehicles_to_plot=vehicles_to_plot,
            scenario_id=scenario_id,
            save_dir=str(save_dir) # Convert Path object to string for older matplotlib if needed
        )

    print(f"Finished processing scenario {scenario_id}. Plotted {len(sorted_timestamps)} timestamps.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Bird\'s-Eye View plots from V2X-ViT dataset YAML files.')
    parser.add_argument('--scenario_id', type=str, required=True, help='Scenario ID (e.g., 2021_08_18_23_23_19)')
    parser.add_argument('--base_dir', type=str, default="dataset/v2x_vit/train", help='Base directory of the dataset (default: dataset/v2x_vit/train)')
    parser.add_argument('--save_dir_base', type=str, default="bev_plots_v2xvit", help='Base directory to save plots (default: bev_plots_v2xvit)')
    args = parser.parse_args()

    process_scenario(
        scenario_id=args.scenario_id,
        base_data_dir=args.base_dir,
        save_dir_base=args.save_dir_base
    )