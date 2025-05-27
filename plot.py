import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# (load_and_flatten_results 函数保持不变)
def load_and_flatten_results(json_filepath):
    if not os.path.exists(json_filepath):
        print(f"Error: JSON file not found at {json_filepath}")
        return pd.DataFrame()

    with open(json_filepath, 'r') as f:
        aggregated_results = json.load(f)

    plot_data_list = []
    metrics_to_extract = ["rmse", "mean", "median", "std", "min", "max"]

    for scenario_name, scenario_data in aggregated_results.items():
        if "sequence_group_results" not in scenario_data:
            continue
        for group_id, group_data in scenario_data["sequence_group_results"].items():
            if "evo_results" not in group_data:
                continue
            for vehicle_case, evo_stats in group_data["evo_results"].items():
                if evo_stats: 
                    record = {
                        "scenario_name": scenario_name,
                        "timestamp_group_id": group_id,
                        "vehicle_case": vehicle_case,
                    }
                    for metric in metrics_to_extract:
                        record[metric] = evo_stats.get(metric)
                    
                    if record.get("rmse") is not None:
                        plot_data_list.append(record)
    
    return pd.DataFrame(plot_data_list)

# (plot_metric_distribution_boxplot 和 plot_metric_cdf 函数保持不变)
def plot_metric_distribution_boxplot(df, metric_name, output_filename="metric_distribution_boxplot.png"):
    if df.empty or metric_name not in df.columns:
        print(f"DataFrame is empty or metric '{metric_name}' not found. Cannot generate box plot.")
        return
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
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_filename); print(f"APE {title_metric_name} distribution box plot saved to: {output_filename}"); plt.close()

def plot_metric_cdf(df, metric_name, output_filename="metric_cdf_plot.png"):
    if df.empty or metric_name not in df.columns:
        print(f"DataFrame is empty or metric '{metric_name}' not found. Cannot generate CDF plot.")
        return
    plot_df = df.dropna(subset=[metric_name])
    if plot_df.empty:
        print(f"No valid data for metric '{metric_name}' after dropping NaNs. Cannot generate CDF plot.")
        return

    plt.figure(figsize=(10, 6))
    vehicle_cases = sorted(plot_df["vehicle_case"].unique())
    palette = sns.color_palette("Set2", len(vehicle_cases)) 
    for i, case in enumerate(vehicle_cases):
        case_values = plot_df[plot_df["vehicle_case"] == case][metric_name].sort_values()
        if case_values.empty: continue
        y_cdf = np.arange(1, len(case_values) + 1) / len(case_values)
        plt.plot(case_values, y_cdf, marker='.', linestyle='-', label=case, color=palette[i % len(palette)], drawstyle='steps-post')
    title_metric_name = metric_name.upper()
    plt.title(f"Cumulative Distribution Function (CDF) of APE {title_metric_name}", fontsize=16)
    plt.xlabel(f"APE {title_metric_name}", fontsize=14); plt.ylabel("Cumulative Probability", fontsize=14)
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    if len(plt.gca().get_lines()) > 0: plt.legend(fontsize=12) # Show legend only if lines were plotted
    plt.grid(True, linestyle='--', alpha=0.7); plt.ylim(0, 1.05)
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_filename); print(f"APE {title_metric_name} CDF plot saved to: {output_filename}"); plt.close()

# --- 新增绘制RMSE百分比变化分布的函数 ---
def plot_rmse_percentage_change_distribution(df_all_results, one_vehicle_case_name="1_vehicle", two_vehicle_case_name="2_vehicles", output_filename="rmse_percentage_change_dist.png"):
    """
    Calculates and plots the distribution of RMSE percentage change 
    from one_vehicle_case to two_vehicle_case.
    """
    if df_all_results.empty:
        print("DataFrame is empty. Cannot calculate or plot RMSE percentage change.")
        return

    # Prepare data for comparison
    df_1_veh = df_all_results[df_all_results['vehicle_case'] == one_vehicle_case_name][
        ['scenario_name', 'timestamp_group_id', 'rmse']
    ].rename(columns={'rmse': 'rmse_1'})

    df_2_veh = df_all_results[df_all_results['vehicle_case'] == two_vehicle_case_name][
        ['scenario_name', 'timestamp_group_id', 'rmse']
    ].rename(columns={'rmse': 'rmse_2'})

    if df_1_veh.empty:
        print(f"No data found for '{one_vehicle_case_name}'. Cannot calculate percentage change.")
        return
    if df_2_veh.empty:
        print(f"No data found for '{two_vehicle_case_name}'. Cannot calculate percentage change.")
        return

    # Merge to get pairs for comparison
    df_comparison = pd.merge(df_1_veh, df_2_veh, on=['scenario_name', 'timestamp_group_id'], how='inner')

    if df_comparison.empty:
        print(f"No matching (scenario, group_id) pairs found between '{one_vehicle_case_name}' and '{two_vehicle_case_name}'. Cannot plot percentage change.")
        return

    # Calculate percentage change: (rmse_2 - rmse_1) / rmse_1 * 100
    # Handle division by zero for rmse_1
    df_comparison['rmse_percent_change'] = np.where(
        df_comparison['rmse_1'] != 0,
        ((df_comparison['rmse_2'] - df_comparison['rmse_1']) / df_comparison['rmse_1']) * 100,
        np.nan # Or a large number if appropriate, or 0 if rmse_1 and rmse_2 are both 0
    )
    
    # Drop rows where percentage change could not be calculated (e.g., rmse_1 was 0, or one of the RMSEs was NaN initially)
    df_comparison.dropna(subset=['rmse_percent_change'], inplace=True)

    if df_comparison.empty:
        print("No valid data points after calculating RMSE percentage change. Cannot plot.")
        return
    
    print(f"\nCalculated RMSE Percentage Change (from '{one_vehicle_case_name}' to '{two_vehicle_case_name}'):")
    print(df_comparison[['scenario_name', 'timestamp_group_id', 'rmse_1', 'rmse_2', 'rmse_percent_change']].head())


    # Plotting the distribution of rmse_percent_change
    plt.figure(figsize=(10, 6))
    sns.histplot(df_comparison['rmse_percent_change'], kde=True, bins=20, color=sns.color_palette("Set2")[2]) # Use a color from the palette
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5) # Add a vertical line at 0% change
    
    plt.title(f"Distribution of RMSE % Change ({one_vehicle_case_name} to {two_vehicle_case_name})", fontsize=16)
    plt.xlabel("RMSE Percentage Change (%)", fontsize=14)
    plt.ylabel("Frequency / Density", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add text for mean/median if desired
    mean_change = df_comparison['rmse_percent_change'].mean()
    median_change = df_comparison['rmse_percent_change'].median()
    plt.text(0.95, 0.90, f"Mean: {mean_change:.2f}%\nMedian: {median_change:.2f}%", 
             horizontalalignment='right', verticalalignment='top', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_filename)
    print(f"RMSE percentage change distribution plot saved to: {output_filename}")
    plt.close()


# --- Main execution ---
if __name__ == "__main__":
    results_json_path = "all_aggregated_evo_ape_results.json" 
    plot_output_base_dir = "plot_result" 

    os.makedirs(plot_output_base_dir, exist_ok=True)
    results_df = load_and_flatten_results(results_json_path)

    if not results_df.empty:
        print("\nDataFrame created from JSON results:")
        print(results_df.head())
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
        
        # --- Call the new plotting function for RMSE percentage change ---
        # Assuming the N-vehicle case you want to compare with "1_vehicle" is named "2_vehicles"
        # If your N-vehicle case has a different name (e.g., "3_vehicles" from len()), adjust "two_vehicle_case_name"
        one_vehicle_case_label = "1_vehicle"
        
        # Determine the N-vehicle case label dynamically if it's not always "2_vehicles"
        # This finds the case with the most vehicles, assuming it's not "1_vehicle"
        n_vehicle_cases = [vc for vc in results_df["vehicle_case"].unique() if vc != one_vehicle_case_label]
        if n_vehicle_cases: # If there are N-vehicle cases
            # Heuristic: pick the one with the largest number in its name if there are multiple N-vehicle cases
            # e.g., if you have "2_vehicles", "3_vehicles", it would pick "3_vehicles".
            # Or, more simply, if you usually only have one type of N-vehicle case (e.g., always "2_vehicles" or always "X_vehicles")
            # you can hardcode it or pick the first one found.
            
            # For simplicity, let's assume there's a dominant N-vehicle case, or you want to compare against "2_vehicles" if it exists.
            # If you always have a case named, for example, based on len(all_vehicles_in_scenario_config),
            # you'd need to ensure that exact name is used.
            # For now, let's target "2_vehicles" as per your formula `rmse_2` for "2车".
            # If "2_vehicles" isn't present, this specific plot won't be generated by the function.
            
            two_vehicle_case_label = "2_vehicles" # Default to "2_vehicles" as per your example
            # More robust: find any case that is not "1_vehicle" if only one other type exists.
            if len(n_vehicle_cases) == 1:
                two_vehicle_case_label = n_vehicle_cases[0]
            elif "2_vehicles" in n_vehicle_cases: # Prioritize "2_vehicles" if it exists among others
                 two_vehicle_case_label = "2_vehicles"
            elif n_vehicle_cases: # Fallback to the first N-vehicle case found if "2_vehicles" is not there
                two_vehicle_case_label = n_vehicle_cases[0]
                print(f"Warning: '{two_vehicle_case_label}' will be used as the N-vehicle case for RMSE comparison.")


            print(f"\nGenerating RMSE percentage change plot (comparing '{one_vehicle_case_label}' with '{two_vehicle_case_label}')...")
            percent_change_plot_filename = os.path.join(plot_output_base_dir, f"ape_rmse_percent_change_dist_{one_vehicle_case_label}_vs_{two_vehicle_case_label}.png")
            plot_rmse_percentage_change_distribution(results_df, 
                                                     one_vehicle_case_name=one_vehicle_case_label,
                                                     two_vehicle_case_name=two_vehicle_case_label,
                                                     output_filename=percent_change_plot_filename)
        else:
            print("\nNo N-vehicle cases (N>1) found to compare with '1_vehicle' for RMSE percentage change plot.")

    else:
        print(f"Could not generate plots because no data was loaded from '{results_json_path}'.")