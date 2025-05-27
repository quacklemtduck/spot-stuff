import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plot style
sns.set_style("whitegrid")

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

# Create boxplot visualization with data points
def plot_position_error_boxplot(df, metric="avg_position_error", 
                                title="Average Position Error Distribution Across Model Versions", 
                                yLabel="Average Position Error",
                                output_file="position_error_boxplot.png"):
    plt.figure(figsize=(12, 8))
    
    # Extract unique dates from column names
    all_dates = set()
    for col in df.columns:
        if metric in col:
            date_part = col.split('_')[0]
            all_dates.add(date_part)
    
    # Sort dates chronologically
    dates = sorted(list(all_dates))
    
    date_alias = {
        '2025-04-09': "Model 1",
        '2025-04-22': "Model 2", 
        '2025-04-30': "Model 3",
        '2025-05-06': "Model 4",
        '2025-05-07': "Model 5",
        '2025-05-13': "Model 6"
    }
    
    # Prepare data for boxplot
    boxplot_data = []
    labels = []
    all_data_points = []
    
    for i, date in enumerate(dates):
        # Find the column with this date and metric
        matching_cols = [col for col in df.columns if date in col and metric in col]
        if matching_cols:
            # Get all values for this date/metric across all commands
            values = df[matching_cols[0]].dropna().values
            boxplot_data.append(values)
            labels.append(date_alias.get(date, date))
            all_data_points.append((i + 1, values))  # Store position and values
    
    # Create boxplot
    box_plot = plt.boxplot(boxplot_data, labels=labels, patch_artist=True)
    
    # Customize boxplot colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize other elements
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black')
    
    # Add individual data points
    for pos, values in all_data_points:
        # Add some jitter to x-coordinates to avoid overlapping
        x_jitter = np.random.normal(pos, 0.04, size=len(values))
        plt.scatter(x_jitter, values, alpha=0.6, s=30, color='red', 
                   edgecolors='darkred', linewidth=0.5, zorder=3)
    
    # Formatting
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Model Version", fontsize=14, labelpad=10)
    plt.ylabel(yLabel, fontsize=14, labelpad=10)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Tight layout to prevent clipping
    plt.tight_layout()
    
    # Always save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")
    
    plt.close()  # Ensure we close the figure

# Seaborn version with data points (using stripplot)
def plot_position_error_boxplot_seaborn(df, metric="avg_position_error", 
                                        title="Average Position Error Distribution Across Model Versions", 
                                        yLabel="Average Position Error",
                                        output_file="position_error_boxplot_seaborn.png",
                                        use_swarm=False):
    plt.figure(figsize=(12, 8))
    
    # Extract unique dates from column names
    all_dates = set()
    for col in df.columns:
        if metric in col:
            date_part = col.split('_')[0]
            all_dates.add(date_part)
    
    # Sort dates chronologically
    dates = sorted(list(all_dates))
    
    date_alias = {
        '2025-04-09': "Model 1",
        '2025-04-22': "Model 2",
        '2025-04-30': "Model 3", 
        '2025-05-06': "Model 4",
        '2025-05-07': "Model 5",
        '2025-05-13': "Model 6"
    }
    
    # Prepare data for seaborn boxplot
    plot_data = []
    
    for date in dates:
        # Find the column with this date and metric
        matching_cols = [col for col in df.columns if date in col and metric in col]
        if matching_cols:
            # Get all values for this date/metric across all commands
            values = df[matching_cols[0]].dropna().values
            for value in values:
                plot_data.append({
                    'Model Version': date_alias.get(date, date),
                    'Value': value
                })
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create boxplot using seaborn
    sns.boxplot(data=plot_df, x='Model Version', y='Value', 
                palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    # Add individual data points
    if use_swarm:
        # Swarmplot tries to avoid overlapping points but may be slow with many points
        sns.swarmplot(data=plot_df, x='Model Version', y='Value', 
                     color='red', alpha=0.6, size=4)
    else:
        # Stripplot is faster and adds jitter
        sns.stripplot(data=plot_df, x='Model Version', y='Value', 
                     color='red', alpha=0.6, size=4, jitter=True)
    
    # Formatting
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Model Version", fontsize=14, labelpad=10)
    plt.ylabel(yLabel, fontsize=14, labelpad=10)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Tight layout to prevent clipping
    plt.tight_layout()
    
    # Always save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")
    
    plt.close()  # Ensure we close the figure

# Enhanced seaborn version with violin plots + points
def plot_position_error_violin(df, metric="avg_position_error", 
                              title="Average Position Error Distribution Across Model Versions", 
                              yLabel="Average Position Error",
                              output_file="position_error_violin.png"):
    plt.figure(figsize=(12, 8))
    
    # Extract unique dates from column names
    all_dates = set()
    for col in df.columns:
        if metric in col:
            date_part = col.split('_')[0]
            all_dates.add(date_part)
    
    # Sort dates chronologically
    dates = sorted(list(all_dates))
    
    date_alias = {
        '2025-04-09': "Model 1",
        '2025-04-22': "Model 2",
        '2025-04-30': "Model 3", 
        '2025-05-06': "Model 4",
        '2025-05-07': "Model 5",
        '2025-05-13': "Model 6"
    }
    
    # Prepare data for seaborn violin plot
    plot_data = []
    
    for date in dates:
        # Find the column with this date and metric
        matching_cols = [col for col in df.columns if date in col and metric in col]
        if matching_cols:
            # Get all values for this date/metric across all commands
            values = df[matching_cols[0]].dropna().values
            for value in values:
                plot_data.append({
                    'Model Version': date_alias.get(date, date),
                    'Value': value
                })
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create violin plot with boxplot inside
    sns.violinplot(data=plot_df, x='Model Version', y='Value', 
                  palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
                  inner='box')
    
    # Add individual data points
    sns.stripplot(data=plot_df, x='Model Version', y='Value', 
                 color='red', alpha=0.6, size=3, jitter=True)
    
    # Formatting
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Model Version", fontsize=14, labelpad=10)
    plt.ylabel(yLabel, fontsize=14, labelpad=10)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Tight layout to prevent clipping
    plt.tight_layout()
    
    # Always save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")
    
    plt.close()  # Ensure we close the figure

def make_plot(file_path: str, metric, title, yLabel, plot_type="boxplot", use_seaborn=True):
    # Make sure the file exists before proceeding
    if os.path.exists(file_path):
        df = load_data(file_path)
        
        if plot_type == "violin":
            output = f"{file_path.removesuffix('.csv')}_{metric}_violin.png"
            plot_position_error_violin(df, metric, title, yLabel, output_file=output)
        else:  # boxplot
            output = f"{file_path.removesuffix('.csv')}_{metric}_boxplot.png"
            if use_seaborn:
                plot_position_error_boxplot_seaborn(df, metric, title, yLabel, output_file=output)
            else:
                plot_position_error_boxplot(df, metric, title, yLabel, output_file=output)
    else:
        print(f"Error: File '{file_path}' not found.")
        print("Please save your data as 'model_comparison_data.csv' or update the file path.")

if __name__ == "__main__":
    files = ["combined_log_0.csv", "combined_log_1.csv", "combined_log_2.csv", 
             "combined_log_3.csv", "combined_log_4.csv", "combined_log_5.csv"]
    metrics = ["avg_position_error", "avg_orientation_error", "position_error_rate", 
               "orientation_error_rate", "position_r_squared", "orientation_r_squared", 
               "max_position_error", "min_position_error", "max_orientation_error", 
               "min_orientation_error", "position_error_pct_change", 
               "orientation_error_pct_change"]
    names = ["Average Position Error", "Average Orientation Error", 
             "Position Error Rate", "Orientation Error Rate", 
             "Position Rate Squared", "Orientation Rate Squared", 
             "Max Position Error", "Min Position Error", 
             "Max Orientation Error", "Min Orientation Error", 
             "Position Error Percentage Change", "Orientation Error Percentage Change"]
    
    # Update this to your data file path
    for f in files:
        for i, m in enumerate(metrics):
            # You can choose "boxplot" or "violin" for plot_type
            make_plot(f, m, f"{names[i]} Distribution Across Model Versions", names[i], 
                     plot_type="violin", use_seaborn=True)
