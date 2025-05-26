import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np # For CDF calculation

def load_and_flatten_results(json_filepath):
    if not os.path.exists(json_filepath):
        print(f"Error: JSON file not found at {json_filepath}")
        return pd.DataFrame()

    with open(json_filepath, 'r') as f:
        aggregated_results = json.load(f)

    plot_data_list = []
    for scenario_name, scenario_data in aggregated_results.items():
        if "sequence_group_results" not in scenario_data:
            continue
        for group_id, group_data in scenario_data["sequence_group_results"].items():
            if "evo_results" not in group_data:
                continue
            for vehicle_case, evo_stats in group_data["evo_results"].items():
                if evo_stats and "rmse" in evo_stats:
                    record = {
                        "scenario_name": scenario_name,
                        "timestamp_group_id": group_id,
                        "vehicle_case": vehicle_case,
                        "rmse": evo_stats["rmse"]
                    }
                    plot_data_list.append(record)
    return pd.DataFrame(plot_data_list)

def plot_rmse_distribution(df, output_filename="rmse_distribution_boxplot.png"):
    if df.empty:
        print("DataFrame is empty. Cannot generate box plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="vehicle_case", y="rmse", data=df, palette="Set2")
    sns.stripplot(x="vehicle_case", y="rmse", data=df, color="0.25", size=4, jitter=True)
    plt.title("Distribution of APE RMSE by Vehicle Case", fontsize=16)
    plt.xlabel("Vehicle Case", fontsize=14)
    plt.ylabel("APE RMSE", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_filename)
    print(f"RMSE distribution box plot saved to: {output_filename}")
    # plt.show() # Comment out if not needed immediately
    plt.close()

def plot_rmse_cdf(df, output_filename="rmse_cdf_plot.png"):
    """
    Generates and saves a CDF plot of RMSE distributions grouped by vehicle_case.
    """
    if df.empty:
        print("DataFrame is empty. Cannot generate CDF plot.")
        return

    plt.figure(figsize=(10, 6))
    
    vehicle_cases = df["vehicle_case"].unique()
    colors = sns.color_palette("Set2", len(vehicle_cases)) # Get distinct colors

    for i, case in enumerate(vehicle_cases):
        # Filter data for the current vehicle case
        case_rmses = df[df["vehicle_case"] == case]["rmse"].sort_values()
        
        if case_rmses.empty:
            print(f"No RMSE data for vehicle case: {case}. Skipping its CDF.")
            continue
            
        # Calculate CDF values
        # Y-axis: cumulative probability (from 0 to 1)
        # For N data points, the probability for the k-th point (0-indexed) is (k+1)/N
        y_cdf = np.arange(1, len(case_rmses) + 1) / len(case_rmses)
        
        # Plot the CDF
        plt.plot(case_rmses, y_cdf, marker='.', linestyle='-', label=case, color=colors[i], drawstyle='steps-post')
        # 'steps-post' drawstyle is common for empirical CDFs

    plt.title("Cumulative Distribution Function (CDF) of APE RMSE", fontsize=16)
    plt.xlabel("APE RMSE", fontsize=14)
    plt.ylabel("Cumulative Probability", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05) # Ensure y-axis goes from 0 to 1 (or slightly above)
    # You might want to set xlim based on your data range, e.g., plt.xlim(0, df["rmse"].max() * 1.1)
    
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_filename)
    print(f"RMSE CDF plot saved to: {output_filename}")
    # plt.show() # Comment out if not needed immediately
    plt.close()


# --- Main execution ---
if __name__ == "__main__":
    results_json_path = "all_aggregated_evo_ape_results.json" 

    results_df = load_and_flatten_results(results_json_path)

    if not results_df.empty:
        print("\nDataFrame created from JSON results:")
        print(results_df.head())
        print(f"\nTotal records for plotting: {len(results_df)}")
        
        print("\nUnique vehicle cases found:")
        print(results_df["vehicle_case"].value_counts())

        # Generate and save the box plot
        plot_rmse_distribution(results_df, output_filename="evo_ape_rmse_distribution_boxplot.png")
        
        # Generate and save the CDF plot
        plot_rmse_cdf(results_df, output_filename="evo_ape_rmse_cdf_plot.png")
    else:
        print(f"Could not generate plots because no data was loaded from '{results_json_path}'.")