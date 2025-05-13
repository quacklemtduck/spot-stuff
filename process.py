import pandas as pd
import os
import re
import argparse
from datetime import datetime
from collections import OrderedDict

def collect_processed_files(base_dir, output_dir):
    """
    Collect all processed metrics files and combine them by log number,
    with commands as rows and folders (sorted by name) as columns.
    
    Parameters:
    base_dir (str): Base directory to search for processed files
    output_dir (str): Directory to save combined files
    
    Returns:
    dict: Statistics about processed files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all processed metrics files recursively
    print(f"Searching for processed metrics files in {base_dir}...")
    all_files = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith('proc_metrics_log_') and file.endswith('.csv'):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        print(f"No processed metrics files found in {base_dir}")
        return {}
    
    print(f"Found {len(all_files)} processed files")
    
    # Group files by log number
    log_groups = {}
    pattern = re.compile(r'proc_metrics_log_(\d+)')
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        match = pattern.search(filename)
        
        if match:
            log_number = match.group(1)
            if log_number not in log_groups:
                log_groups[log_number] = []
            log_groups[log_number].append(file_path)
    
    print(f"Grouped files into {len(log_groups)} log numbers")
    
    # Process each group
    stats = {
        'total_files': len(all_files),
        'groups': len(log_groups),
        'combined_files': []
    }
    
    for log_number, file_paths in log_groups.items():
        print(f"\nProcessing log group {log_number} with {len(file_paths)} files")
        
        # First, identify all unique folder names and sort them
        folder_names = []
        file_paths_by_folder = {}
        
        for file_path in file_paths:
            folder_name = os.path.basename(os.path.dirname(file_path))
            if folder_name not in folder_names:
                folder_names.append(folder_name)
                file_paths_by_folder[folder_name] = file_path
        
        # Sort folder names alphabetically
        folder_names.sort()
        print(f"  Found {len(folder_names)} unique folders: {folder_names}")
        
        # Dictionary to hold data from each folder
        folder_data = {}
        
        # Columns to use for the combined data
        metrics_columns = [
            'avg_position_error', 
            'avg_orientation_error', 
            'position_error_rate', 
            'orientation_error_rate',
            'position_r_squared',
            'orientation_r_squared',
            'max_position_error',
            'min_position_error',
            'max_orientation_error',
            'min_orientation_error',
            'position_error_pct_change',
            'orientation_error_pct_change',
            'num_steps'
        ]
        
        # Process each folder in sorted order
        for folder_name in folder_names:
            file_path = file_paths_by_folder[folder_name]
            try:
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                print(f"  Processing {folder_name} from {file_path}")
                
                # Check if the file has the expected structure
                if 'command' not in df.columns:
                    print(f"  Warning: {file_path} doesn't have a 'command' column, skipping")
                    continue
                
                # Store data by command
                for _, row in df.iterrows():
                    command = row['command']
                    
                    if command not in folder_data:
                        folder_data[command] = OrderedDict()
                    
                    # Add metrics for this folder
                    for metric in metrics_columns:
                        if metric in row:
                            column_name = f"{folder_name}_{metric}"
                            folder_data[command][column_name] = row[metric]
            
            except Exception as e:
                print(f"  Error processing {file_path}: {str(e)}")
        
        if folder_data:
            # Convert to DataFrame
            combined_df = pd.DataFrame.from_dict(folder_data, orient='index')
            combined_df.index.name = 'command'
            combined_df.reset_index(inplace=True)
            
            # Sort commands numerically if possible
            try:
                combined_df['command'] = pd.to_numeric(combined_df['command']).astype(int)
                combined_df = combined_df.sort_values('command')
            except:
                # If commands can't be converted to numbers, sort them as strings
                combined_df = combined_df.sort_values('command')
            
            # Ensure columns are in the correct order (sorted by folder name)
            columns = ['command']
            for folder_name in folder_names:
                for metric in metrics_columns:
                    column_name = f"{folder_name}_{metric}"
                    if column_name in combined_df.columns:
                        columns.append(column_name)
            
            # Reorder columns
            combined_df = combined_df[columns]
            
            # Create output filename
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_filename = f"combined_log_{log_number}_{timestamp}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save combined data
            combined_df.to_csv(output_path, index=False)
            print(f"  Saved combined data to {output_path} with {len(combined_df)} commands")
            
            stats['combined_files'].append({
                'log_number': log_number,
                'files_combined': len(file_paths),
                'commands_combined': len(folder_data),
                'output_file': output_path
            })
    
    return stats

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Collect and combine processed metrics files by log number.')
    parser.add_argument('base_dir', type=str, help='Base directory to search for processed files')
    parser.add_argument('--output-dir', type=str, default='combined_results',
                       help='Directory to save combined files (default: "combined_results")')
    
    args = parser.parse_args()
    
    # Check if base directory exists
    if not os.path.isdir(args.base_dir):
        print(f"Error: {args.base_dir} is not a valid directory")
        return 1
    
    # Process files
    stats = collect_processed_files(args.base_dir, args.output_dir)
    
    if not stats:
        return 1
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Number of log groups: {stats['groups']}")
    print(f"Combined files created: {len(stats['combined_files'])}")
    
    for combined_file in stats['combined_files']:
        print(f"\nLog {combined_file['log_number']}:")
        print(f"  Files combined: {combined_file['files_combined']}")
        print(f"  Commands combined: {combined_file['commands_combined']}")
        print(f"  Output file: {combined_file['output_file']}")
    
    return 0

if __name__ == "__main__":
    exit(main())
