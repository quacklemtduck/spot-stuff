import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plot style
sns.set_style("whitegrid")

# Load data from file with proper missing data handling
def load_data(file_path):
    # Read the data
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get header and data
    header = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:]]
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=header)
    
    # Replace empty strings with NaN
    df = df.replace('', np.nan)
    
    # Convert command to int, handling missing values
    df['command'] = pd.to_numeric(df['command'], errors='coerce').astype('Int64')
    
    # Convert other columns to float, handling missing values
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows where command is NaN (invalid command entries)
    df = df.dropna(subset=['command'])
    
    # Set command as index
    df.set_index('command', inplace=True)
    
    return df

# Create boxplot visualization with distinguishable data points
def plot_position_error_boxplot(df, metric="avg_position_error", 
                                title="Average Position Error Distribution Across Model Versions", 
                                yLabel="Average Position Error",
                                output_file="position_error_boxplot.png"):
    plt.figure(figsize=(14, 9))
    
    # Extract unique dates from column names
    all_dates = set()
    for col in df.columns:
        if metric in col:
            date_part = col.split('_')[0]
            all_dates.add(date_part)
    
    # Sort dates chronologically
    dates = sorted(list(all_dates))
    
    # Check if we have any data for this metric
    if not dates:
        print(f"No data found for metric: {metric}")
        plt.close()
        return
    
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
            values_with_commands = []
            for command in df.index:
                value = df.loc[command, matching_cols[0]]
                if pd.notna(value):  # Only include non-NaN values
                    values_with_commands.append((command, value))
            
            if values_with_commands:  # Only proceed if we have data
                values = [v[1] for v in values_with_commands]
                
                boxplot_data.append(values)
                labels.append(date_alias.get(date, date))
                all_data_points.append((i + 1, values_with_commands))
    
    # Check if we have any data to plot
    if not boxplot_data:
        print(f"No valid data found for metric: {metric}")
        plt.close()
        return
    
    # Create boxplot with higher transparency and lower z-order
    box_plot = plt.boxplot(boxplot_data, labels=labels, patch_artist=True)
    
    # Customize boxplot colors with higher transparency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)  # More transparent
        patch.set_zorder(1)   # Lower z-order
    
    # Customize other elements with lower z-order
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box_plot[element], color='black', zorder=1)
    
    # Define colors and markers for different commands
    unique_commands = sorted([cmd for cmd in df.index if pd.notna(cmd)])
    command_colors = plt.cm.tab10(np.arange(len(unique_commands)))
    markers = ['o', 's', '^', 'v', 'd', '*', 'p', 'h', 'X', '+']
    
    # Add individual data points without jitter with high z-order
    plotted_commands = set()
    for pos, values_with_commands in all_data_points:
        for command, value in values_with_commands:
            if pd.notna(command) and pd.notna(value):  # Double-check for valid data
                cmd_idx = unique_commands.index(command) if command in unique_commands else 0
                color = command_colors[cmd_idx]
                marker = markers[cmd_idx % len(markers)]
                
                # Only add to legend once per command
                label = f'Command {int(command)}' if command not in plotted_commands else None
                plotted_commands.add(command)
                
                plt.scatter(pos, value, alpha=0.9, s=80, color=color, 
                           marker=marker, edgecolors='black', linewidth=1, 
                           zorder=10, label=label)  # High z-order to appear on top
    
    # Add legend if we have any commands
    if plotted_commands:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    # Formatting
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Model Version", fontsize=14, labelpad=10)
    plt.ylabel(yLabel, fontsize=14, labelpad=10)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Add grid with low z-order
    plt.grid(True, linestyle='--', alpha=0.7, axis='y', zorder=0)
    
    # Tight layout to prevent clipping
    plt.tight_layout()
    
    # Always save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")
    
    plt.close()  # Ensure we close the figure

# Seaborn version with distinguishable data points
def plot_position_error_boxplot_seaborn(df, metric="avg_position_error", 
                                        title="Average Position Error Distribution Across Model Versions", 
                                        yLabel="Average Position Error",
                                        output_file="position_error_boxplot_seaborn.png"):
    plt.figure(figsize=(14, 9))
    
    # Extract unique dates from column names
    all_dates = set()
    for col in df.columns:
        if metric in col:
            date_part = col.split('_')[0]
            all_dates.add(date_part)
    
    # Sort dates chronologically
    dates = sorted(list(all_dates))
    
    # Check if we have any data for this metric
    if not dates:
        print(f"No data found for metric: {metric}")
        plt.close()
        return
    
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
            for command in df.index:
                value = df.loc[command, matching_cols[0]]
                if pd.notna(value) and pd.notna(command):  # Only include valid data
                    plot_data.append({
                        'Model Version': date_alias.get(date, date),
                        'Value': value,
                        'Command': int(command)
                    })
    
    # Check if we have data to plot
    if not plot_data:
        print(f"No valid data found for metric: {metric}")
        plt.close()
        return
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create boxplot using seaborn with transparency
    ax = sns.boxplot(data=plot_df, x='Model Version', y='Value', 
                     palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    # Make boxplot more transparent
    for patch in ax.artists:
        patch.set_alpha(0.6)
    
    # Add individual data points without jitter, colored by command
    unique_commands = sorted(plot_df['Command'].unique())
    command_colors = plt.cm.tab10(np.arange(len(unique_commands)))
    markers = ['o', 's', '^', 'v', 'd', '*', 'p', 'h', 'X', '+']
    
    # Plot points for each command
    for i, command in enumerate(unique_commands):
        command_data = plot_df[plot_df['Command'] == command]
        color = command_colors[i]
        marker = markers[i % len(markers)]
        
        # Get x positions for each model version
        model_versions = command_data['Model Version'].unique()
        for model_version in model_versions:
            model_data = command_data[command_data['Model Version'] == model_version]
            # Get the x position for this model version
            x_pos = plot_df['Model Version'].unique().tolist().index(model_version)
            
            plt.scatter([x_pos] * len(model_data), model_data['Value'], 
                       color=color, marker=marker, s=80, alpha=0.9, 
                       edgecolor='black', linewidth=1, zorder=10,
                       label=f'Command {command}' if model_version == model_versions[0] else "")
    
    # Remove duplicate legend entries
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    if by_label:  # Only add legend if we have entries
        plt.legend(by_label.values(), by_label.keys(), 
                  bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
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

# Enhanced seaborn version with violin plots + distinguishable points
def plot_position_error_violin(df, metric="avg_position_error", 
                              title="Average Position Error Distribution Across Model Versions", 
                              yLabel="Average Position Error",
                              output_file="position_error_violin.png"):
    plt.figure(figsize=(14, 9))
    
    # Extract unique dates from column names
    all_dates = set()
    for col in df.columns:
        if metric in col:
            date_part = col.split('_')[0]
            all_dates.add(date_part)
    
    # Sort dates chronologically
    dates = sorted(list(all_dates))
    
    # Check if we have any data for this metric
    if not dates:
        print(f"No data found for metric: {metric}")
        plt.close()
        return
    
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
            for command in df.index:
                value = df.loc[command, matching_cols[0]]
                if pd.notna(value) and pd.notna(command):  # Only include valid data
                    plot_data.append({
                        'Model Version': date_alias.get(date, date),
                        'Value': value,
                        'Command': int(command)
                    })
    
    # Check if we have data to plot
    if not plot_data:
        print(f"No valid data found for metric: {metric}")
        plt.close()
        return
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create violin plot with boxplot inside and transparency
    ax = sns.violinplot(data=plot_df, x='Model Version', y='Value', 
                       palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
                       inner='box', alpha=0.6)
    
    # Add individual data points without jitter, colored by command
    unique_commands = sorted(plot_df['Command'].unique())
    command_colors = plt.cm.tab10(np.arange(len(unique_commands)))
    markers = ['o', 's', '^', 'v', 'd', '*', 'p', 'h', 'X', '+']
    
    # Plot points for each command
    for i, command in enumerate(unique_commands):
        command_data = plot_df[plot_df['Command'] == command]
        color = command_colors[i]
        marker = markers[i % len(markers)]
        
        # Get x positions for each model version
        model_versions = command_data['Model Version'].unique()
        for model_version in model_versions:
            model_data = command_data[command_data['Model Version'] == model_version]
            # Get the x position for this model version
            x_pos = plot_df['Model Version'].unique().tolist().index(model_version)
            
            plt.scatter([x_pos] * len(model_data), model_data['Value'], 
                       color=color, marker=marker, s=70, alpha=0.9, 
                       edgecolor='black', linewidth=1, zorder=10,
                       label=f'Command {command}' if model_version == model_versions[0] else "")
    
    # Remove duplicate legend entries
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    if by_label:  # Only add legend if we have entries
        plt.legend(by_label.values(), by_label.keys(), 
                  bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
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
        try:
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
        except Exception as e:
            print(f"Error processing {file_path} for metric {metric}: {str(e)}")
    else:
        print(f"Error: File '{file_path}' not found.")

if __name__ == "__main__":
    files = ["combined_log_0.csv", "combined_log_1.csv", "combined_log_2.csv", 
             "combined_log_345.csv"]
    file_names = ["1. Easy", "2. Far away", "3. Close", "4. Sides & Back"]
    # metrics = ["avg_position_error", "avg_orientation_error", "position_error_rate", 
    #            "orientation_error_rate", "position_r_squared", "orientation_r_squared", 
    #            "max_position_error", "min_position_error", "max_orientation_error", 
    #            "min_orientation_error", "position_error_pct_change", 
    #            "orientation_error_pct_change"]
    # names = ["Average Position Error in meters", "Average Orientation Error in radians", 
    #          "Position Error Rate", "Orientation Error Rate", 
    #          "Position Rate Squared", "Orientation Rate Squared", 
    #          "Max Position Error", "Min Position Error", 
    #          "Max Orientation Error", "Min Orientation Error", 
    #          "Position Error Percentage Change", "Orientation Error Percentage Change"]
    metrics = ["avg_position_error", "avg_orientation_error", "min_position_error",  "min_orientation_error" ]
    names = ["Average Position Error (Meters)", "Average Orientation Error (Radians)", "Min Position Error (Meters)", "Min Orientation Error (Radians)"]
    
    # Update this to your data file path
    for j, f in enumerate(files):
        for i, m in enumerate(metrics):
            # You can choose "boxplot" or "violin" for plot_type
            make_plot(f, m, f"{file_names[j]} - {names[i]} Across Model Versions", names[i], 
                     plot_type="boxplot", use_seaborn=True)
