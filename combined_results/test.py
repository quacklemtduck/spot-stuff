import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plot style without specifying Arial font
sns.set_style("whitegrid")
# Use default system fonts instead of specifically requiring Arial
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial']

# Load data from file
def load_data(file_path):
    # Read the data
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get header and data
    header = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:]]
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=header)
    
    # Convert command to int and other columns to float
    df['command'] = df['command'].astype(int)
    for col in df.columns[1:]:
        df[col] = df[col].astype(float)
    
    # Set command as index
    df.set_index('command', inplace=True)
    
    return df

# Create visualization
def plot_position_error_comparison(df, metric = "avg_position_error", 
                                   title="Average Position Error Across Model Versions", 
                                   yLabel="Average Position Error",
                                   output_file="position_error_comparison.png"):
    plt.figure(figsize=(14, 9))
    
    # Extract unique dates from column names
    all_dates = set()
    for col in df.columns:
        if metric in col:
            date_part = col.split('_')[0]
            all_dates.add(date_part)
    
    # Sort dates chronologically
    dates = sorted(list(all_dates))
    
    # Define a better color palette - using tab10 which is colorblind-friendly
    # and provides more distinct colors than viridis
    #colors = plt.cm.tab10(np.arange(len(df.index)))
    
    # Alternative custom color palette (uncomment to use)
    custom_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf"   # teal
    ]
    
    date_alias = {
            '2025-04-09': "Model 1",
            '2025-04-22': "Model 2",
            '2025-04-30': "Model 3",
            '2025-05-06': "Model 4",
            '2025-05-07': "Model 5",
            '2025-05-13': "Model 6"
            }

    markers = ['o', 's', '^', 'v', 'd', '*', 'p', 'h', 'X']
    # For each command, plot position error across dates
    for i, command in enumerate(df.index):
        position_errors = []
        
        for date in dates:
            # Find the column with this date and avg_position_error
            matching_cols = [col for col in df.columns if date in col and metric in col]
            if matching_cols:
                position_errors.append(df.loc[command, matching_cols[0]])
            else:
                position_errors.append(np.nan)  # Handle missing data
        
        marker_shape = markers[i % len(markers)]
        # Changed from solid to dotted lines
        plt.plot(range(len(dates)), position_errors, marker=marker_shape, markersize=8, 
             linewidth=1, linestyle='--', color=custom_colors[i], label=f"Command {command}")
    
    # Formatting
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Model Version", fontsize=14, labelpad=10)
    plt.ylabel(yLabel, fontsize=14, labelpad=10)
    
    # Format x-axis with dates
    plt.xticks(range(len(dates)), [date_alias[date] for date in dates], 
               rotation=45, ha='right', fontsize=12)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Tight layout to prevent clipping
    plt.tight_layout()
    
    # Always save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")
    
    plt.close()  # Ensure we close the figure

def make_plot(file_path: str, metric, title, yLabel):
    # Make sure the file exists before proceeding
    if os.path.exists(file_path):
        df = load_data(file_path)
        output = f"{file_path.removesuffix('.csv')}_{metric}.png"
        plot_position_error_comparison(df, metric, title, yLabel, output_file=output)
    else:
        print(f"Error: File '{file_path}' not found.")
        print("Please save your data as 'model_comparison_data.csv' or update the file path.")

if __name__ == "__main__":
    files = ["combined_log_0.csv", "combined_log_1.csv", "combined_log_2.csv", "combined_log_3.csv", "combined_log_4.csv", "combined_log_5.csv", ]
    metrics = ["avg_position_error", "avg_orientation_error", "position_error_rate", "orientation_error_rate", "position_r_squared", "orientation_r_squared", "max_position_error", "min_position_error", "max_orientation_error", "min_orientation_error", "position_error_pct_change", "orientation_error_pct_change"]
    names = ["Average Position Error", "Average Orientation Error", "Position Error Rate", "Orientation Error Rate", "Position Rate Squared", "Orientation Rate Squared", "Max Position Error", "Min Position Error", "Max Orientation Error", "Min Orientation Error", "Position Error Percentage Change", "Orientation Error Percentage Change"]
    # Update this to your data file path
    for f in files:
        for i, m in enumerate(metrics):
            make_plot(f, m, f"{names[i]} Across Model Versions", names[i])
