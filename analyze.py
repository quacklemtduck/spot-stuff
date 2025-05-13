import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import os
import glob
import sys
from datetime import datetime

def analyze_command_errors(file_path, output_path=None, save_plot=False):
    """
    Analyze command errors from the specified file and return metrics on error
    magnitude and reduction rate.
    
    Parameters:
    file_path (str): Path to the input CSV file
    output_path (str, optional): Path to save the results CSV
    save_plot (bool): Whether to save a plot of the errors
    
    Returns:
    tuple: (results_df, success_flag)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: The file {file_path} does not exist")
            return None, False
        
        # Load the data using pandas
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, header=0, engine='python')
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        print(f"Loaded {len(df)} data points")
        
        # Get unique commands
        commands = df['command_counter'].unique()
        print(f"Found {len(commands)} unique commands: {commands}")
        
        # For storing results
        results = []
        
        # Process each command
        for cmd in commands:
            cmd_data = df[df['command_counter'] == cmd]
            
            # Calculate average errors for this command
            avg_pos_error = cmd_data['position_error'].mean()
            avg_orient_error = cmd_data['orientation_error'].mean()
            
            # Calculate error reduction rates by fitting a line to the errors over steps
            if len(cmd_data) > 1:  # Need at least 2 points for a line
                x = np.array(range(len(cmd_data)))
                
                # Position error slope (negative value = errors decreasing)
                pos_slope, pos_intercept, pos_r_value, _, _ = stats.linregress(x, cmd_data['position_error'])
                
                # Orientation error slope
                orient_slope, orient_intercept, orient_r_value, _, _ = stats.linregress(x, cmd_data['orientation_error'])
                
                # Calculate R-squared values
                pos_r_squared = pos_r_value**2
                orient_r_squared = orient_r_value**2
            else:
                pos_slope = 0
                orient_slope = 0
                pos_r_squared = 0
                orient_r_squared = 0
            
            # Maximum, minimum, and first vs last errors
            max_pos_error = cmd_data['position_error'].max()
            min_pos_error = cmd_data['position_error'].min()
            max_orient_error = cmd_data['orientation_error'].max()
            min_orient_error = cmd_data['orientation_error'].min()
            
            first_pos_error = cmd_data['position_error'].iloc[0]
            last_pos_error = cmd_data['position_error'].iloc[-1]
            first_orient_error = cmd_data['orientation_error'].iloc[0]
            last_orient_error = cmd_data['orientation_error'].iloc[-1]
            
            # Calculate percentage change from first to last
            pos_pct_change = 100 * (last_pos_error - first_pos_error) / first_pos_error if first_pos_error != 0 else 0
            orient_pct_change = 100 * (last_orient_error - first_orient_error) / first_orient_error if first_orient_error != 0 else 0
            
            # Store results
            results.append({
                'command': cmd,
                'num_steps': len(cmd_data),
                'avg_position_error': avg_pos_error,
                'avg_orientation_error': avg_orient_error,
                'position_error_rate': pos_slope,
                'orientation_error_rate': orient_slope,
                'position_r_squared': pos_r_squared,
                'orientation_r_squared': orient_r_squared,
                'max_position_error': max_pos_error,
                'min_position_error': min_pos_error,
                'max_orientation_error': max_orient_error,
                'min_orientation_error': min_orient_error,
                'position_error_pct_change': pos_pct_change,
                'orientation_error_pct_change': orient_pct_change
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results automatically if output_path is provided
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        # Create and save plots if requested
        if save_plot:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 1, 1)
            for cmd in commands:
                cmd_data = df[df['command_counter'] == cmd]
                plt.plot(cmd_data['step'], cmd_data['position_error'], label=f'Command {cmd}')
            plt.title('Position Error by Command')
            plt.xlabel('Step')
            plt.ylabel('Position Error')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            for cmd in commands:
                cmd_data = df[df['command_counter'] == cmd]
                plt.plot(cmd_data['step'], cmd_data['orientation_error'], label=f'Command {cmd}')
            plt.title('Orientation Error by Command')
            plt.xlabel('Step')
            plt.ylabel('Orientation Error')
            plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_path.replace('.csv', '.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
            plt.close()  # Close the figure to free memory
        
        # Return results and success flag
        return results_df, True
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False

def find_metrics_files(base_dir):
    """
    Find all files starting with 'metrics_' in the given directory and its subdirectories.
    
    Parameters:
    base_dir (str): Base directory to search
    
    Returns:
    list: List of matching file paths
    """
    metrics_files = []
    
    # Walk through the directory structure
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith('metrics_') and file.endswith('.csv'):
                full_path = os.path.join(root, file)
                metrics_files.append(full_path)
    
    return metrics_files

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Analyze command errors from metrics files.')
    parser.add_argument('base_dir', type=str, help='Base directory to search for metrics files')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--summary', action='store_true', help='Generate a summary of all processed files')
    
    args = parser.parse_args()
    
    # Check if base directory exists
    if not os.path.isdir(args.base_dir):
        print(f"Error: {args.base_dir} is not a valid directory")
        return 1
    
    # Find all metrics files
    print(f"Searching for metrics files in {args.base_dir}...")
    metrics_files = find_metrics_files(args.base_dir)
    
    if not metrics_files:
        print(f"No metrics files found in {args.base_dir}")
        return 1
    
    print(f"Found {len(metrics_files)} metrics files to process")
    
    # Process each file
    summary_data = []
    success_count = 0
    
    for file_path in metrics_files:
        print(f"\nProcessing {file_path}...")
        
        # Create output path with "proc_" prefix
        input_dir = os.path.dirname(file_path)
        input_filename = os.path.basename(file_path)
        output_filename = f"proc_{input_filename}"
        output_path = os.path.join(input_dir, output_filename)
        
        # Process the file
        results_df, success = analyze_command_errors(
            file_path, 
            output_path=output_path,
            save_plot=not args.no_plots
        )
        
        if success:
            success_count += 1
            
            # Collect summary data if requested
            if args.summary and results_df is not None:
                # Add file information to each command result
                for _, row in results_df.iterrows():
                    summary_row = row.to_dict()
                    summary_row['file_path'] = file_path
                    summary_row['file_name'] = input_filename
                    summary_data.append(summary_row)
    
    # Print processing statistics
    print(f"\nProcessing complete: {success_count} of {len(metrics_files)} files processed successfully")
    
    # Generate and save summary if requested
    if args.summary and summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Create a timestamp for the summary file
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        summary_path = os.path.join(args.base_dir, f"summary_analysis_{timestamp}.csv")
        
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
        
        # Print a quick summary of the summary
        print("\nSummary Statistics by Command:")
        command_summary = summary_df.groupby('command').agg({
            'num_steps': 'mean',
            'avg_position_error': 'mean',
            'avg_orientation_error': 'mean',
            'position_error_rate': 'mean',
            'orientation_error_rate': 'mean'
        })
        print(command_summary)
    
    return 0

if __name__ == "__main__":
    exit(main())
