import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Function to load and flatten results ---
def load_and_flatten_results(json_filepath):
    """
    Loads the aggregated JSON results and flattens them into a Pandas DataFrame.
    Each row will represent one EVO run with its metrics and context.
    """
    if not os.path.exists(json_filepath):
        print(f"Error: JSON file not found at {json_filepath}")
        return pd.DataFrame()

    with open(json_filepath, 'r') as f:
        aggregated_results = json.load(f)

    plot_data_list = []
    metrics_to_extract = ["rmse", "mean", "median", "std", "min", "max"] # sse is usually not plotted this way

    for scenario_name, scenario_data in aggregated_results.items():
        if "sequence_group_results" not in scenario_data:
            continue
        for group_id, group_data in scenario_data["sequence_group_results"].items():
            if "evo_results" not in group_data:
                continue
            for vehicle_case, evo_stats in group_data["evo_results"].items():
                if evo_stats: # Check if evo_stats is not None
                    record = {
                        "scenario_name": scenario_name,
                        "timestamp_group_id": group_id,
                        "vehicle_case": vehicle_case,
                    }
                    for metric in metrics_to_extract:
                        record[metric] = evo_stats.get(metric) # Use .get() for safety
                    
                    # Only append if at least one primary metric (e.g., rmse) is present
                    if record.get("rmse") is not None:
                        plot_data_list.append(record)
                    # else:
                        # print(f"Warning: RMSE missing for {scenario_name}, group {group_id}, case {vehicle_case}")
                # else:
                    # print(f"Warning: Missing evo_stats for {scenario_name}, group {group_id}, case {vehicle_case}")
    
    return pd.DataFrame(plot_data_list)

# --- Generic plotting functions ---
def plot_metric_distribution_boxplot(df, metric_name, output_filename="metric_distribution_boxplot.png"):
    """
    Generates and saves a box plot of a given metric's distributions grouped by vehicle_case.
    """
    if df.empty or metric_name not in df.columns:
        print(f"DataFrame is empty or metric '{metric_name}' not found. Cannot generate box plot.")
        return
    
    # Filter out rows where the metric might be NaN (if .get() returned None and it became NaN)
    plot_df = df.dropna(subset=[metric_name])
    if plot_df.empty:
        print(f"No valid data for metric '{metric_name}' after dropping NaNs. Cannot generate box plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="vehicle_case", y=metric_name, data=plot_df, hue="vehicle_case", palette="Set2", legend=False, dodge=False)
    sns.stripplot(x="vehicle_case", y=metric_name, data=plot_df, color="0.25", size=4, jitter=True)

    title_metric_name = metric_name.upper()
    plt.title(f"Distribution of APE {title_metric_name} by Vehicle Case", fontsize=16)
    plt.xlabel("Vehicle Case", fontsize=14)
    plt.ylabel(f"APE {title_metric_name}", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        
    plt.savefig(output_filename)
    print(f"APE {title_metric_name} distribution box plot saved to: {output_filename}")
    plt.close() # Close plot to free memory

def plot_metric_cdf(df, metric_name, output_filename="metric_cdf_plot.png"):
    """
    Generates and saves a CDF plot of a given metric's distributions grouped by vehicle_case.
    """
    if df.empty or metric_name not in df.columns:
        print(f"DataFrame is empty or metric '{metric_name}' not found. Cannot generate CDF plot.")
        return

    # Filter out rows where the metric might be NaN
    plot_df = df.dropna(subset=[metric_name])
    if plot_df.empty:
        print(f"No valid data for metric '{metric_name}' after dropping NaNs. Cannot generate CDF plot.")
        return

    plt.figure(figsize=(10, 6))
    
    vehicle_cases = sorted(plot_df["vehicle_case"].unique()) # Sort for consistent legend order
    palette = sns.color_palette("Set2", len(vehicle_cases)) 

    for i, case in enumerate(vehicle_cases):
        case_values = plot_df[plot_df["vehicle_case"] == case][metric_name].sort_values()
        
        if case_values.empty:
            # print(f"No data for metric '{metric_name}' in vehicle case: {case}. Skipping its CDF.")
            continue
            
        y_cdf = np.arange(1, len(case_values) + 1) / len(case_values)
        
        plt.plot(case_values, y_cdf, marker='.', linestyle='-', label=case, color=palette[i % len(palette)], drawstyle='steps-post')

    title_metric_name = metric_name.upper()
    plt.title(f"Cumulative Distribution Function (CDF) of APE {title_metric_name}", fontsize=16)
    plt.xlabel(f"APE {title_metric_name}", fontsize=14)
    plt.ylabel("Cumulative Probability", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if len(vehicle_cases) > 0 and not all(plot_df[plot_df["vehicle_case"] == c][metric_name].empty for c in vehicle_cases):
        plt.legend(fontsize=12) # Only show legend if there's something to label
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    # Consider setting xlim based on data range if needed
    # current_min_val = plot_df[metric_name].min()
    # current_max_val = plot_df[metric_name].max()
    # if pd.notna(current_min_val) and pd.notna(current_max_val):
    #     plt.xlim(current_min_val * 0.9, current_max_val * 1.1)


    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        
    plt.savefig(output_filename)
    print(f"APE {title_metric_name} CDF plot saved to: {output_filename}")
    plt.close() # Close plot to free memory

# --- Main execution ---
if __name__ == "__main__":
    results_json_path = "all_aggregated_evo_ape_results.json" 
    plot_output_base_dir = "plot_result" # Define the output folder for plots

    # Create the base directory for plots if it doesn't exist
    os.makedirs(plot_output_base_dir, exist_ok=True)

    results_df = load_and_flatten_results(results_json_path)

    if not results_df.empty:
        print("\nDataFrame created from JSON results:")
        print(results_df.head())
        # print(f"\nDataFrame Info:") # For debugging data types and non-null counts
        # results_df.info()
        print(f"\nTotal records for plotting: {len(results_df)}")
        
        print("\nUnique vehicle cases found:")
        print(results_df["vehicle_case"].value_counts())

        metrics_to_plot = ["rmse", "mean", "median", "std", "min", "max"]

        for metric in metrics_to_plot:
            print(f"\nGenerating plots for metric: {metric.upper()}...")
            
            boxplot_filename = os.path.join(plot_output_base_dir, f"ape_{metric}_distribution_boxplot.png")
            plot_metric_distribution_boxplot(results_df, metric, output_filename=boxplot_filename)
            
            cdf_plot_filename = os.path.join(plot_output_base_dir, f"ape_{metric}_cdf_plot.png")
            plot_metric_cdf(results_df, metric, output_filename=cdf_plot_filename)
    else:
        print(f"Could not generate plots because no data was loaded from '{results_json_path}'.")