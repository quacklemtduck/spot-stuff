import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import os

def analyze_command_errors(file_path, output_path=None):
    """
    Analyze command errors from the specified file and return metrics on error
    magnitude and reduction rate.
    
    Parameters:
    file_path (str): Path to the input CSV file
    output_path (str, optional): Path to save the results CSV
    
    Returns:
    tuple: (results_df, figure)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")
    
    # Load the data using pandas
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=0, engine='python')
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    print(f"Loaded {len(df)} data points")
    print(f"Columns in the data (after stripping whitespace): {df.columns.tolist()}")
    
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
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    # Create plots
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
    
    # Return results and figure
    return results_df, plt.gcf()

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Analyze command errors from a data file.')
    parser.add_argument('file_path', type=str, help='Path to the data file')
    parser.add_argument('--output', type=str, default=None, 
                        help='Path to save results CSV (optional, defaults to "proc_<original_filename>")')
    parser.add_argument('--no-plot', action='store_true', 
                        help='Disable plotting')
    parser.add_argument('--plot-output', type=str, default=None,
                        help='Path to save plot image (optional)')
    
    args = parser.parse_args()
    
    # Create default output path if not specified
    if args.output is None:
        # Get the directory and filename from the input path
        input_dir = os.path.dirname(args.file_path)
        input_filename = os.path.basename(args.file_path)
        
        # Create the output filename with "proc_" prefix
        output_filename = f"proc_{input_filename}"
        
        # Create the full output path
        args.output = os.path.join(input_dir, output_filename)
        print(f"Will save results to: {args.output}")
    
    # Run analysis
    try:
        results_df, fig = analyze_command_errors(args.file_path, args.output)
        
        # Print results
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)  # Increased width for better display
        pd.set_option('display.precision', 4)  # Limit decimal places for better readability
        print("\nAnalysis Results:")
        print(results_df)
        
        # Show or save plot
        if not args.no_plot:
            if args.plot_output:
                fig.savefig(args.plot_output, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {args.plot_output}")
            else:
                plt.show()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
