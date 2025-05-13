import pandas as pd
import os
import glob
import re
import argparse
from datetime import datetime

def collect_processed_files(base_dir, output_dir):
    """
    Collect all processed metrics files and combine them by log number.
    
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
        
        # Combine all files in this group
        combined_data = []
        
        for file_path in file_paths:
            try:
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                # Add source information
                folder_name = os.path.basename(os.path.dirname(file_path))
                df['source_folder'] = folder_name
                df['source_file'] = os.path.basename(file_path)
                
                combined_data.append(df)
                print(f"  Added {file_path}")
            except Exception as e:
                print(f"  Error processing {file_path}: {str(e)}")
        
        if combined_data:
            # Combine all DataFrames
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Create output filename
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_filename = f"combined_log_{log_number}_{timestamp}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save combined data
            combined_df.to_csv(output_path, index=False)
            print(f"  Saved combined data to {output_path}")
            
            # Generate summary statistics
            summary_by_folder = combined_df.groupby(['source_folder', 'command']).agg({
                'avg_position_error': 'mean',
                'avg_orientation_error': 'mean',
                'position_error_rate': 'mean',
                'orientation_error_rate': 'mean'
            })
            
            summary_filename = f"summary_log_{log_number}_{timestamp}.csv"
            summary_path = os.path.join(output_dir, summary_filename)
            summary_by_folder.to_csv(summary_path)
            print(f"  Saved summary to {summary_path}")
            
            stats['combined_files'].append({
                'log_number': log_number,
                'files_combined': len(file_paths),
                'output_file': output_path,
                'summary_file': summary_path
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
        print(f"  Output file: {combined_file['output_file']}")
        print(f"  Summary file: {combined_file['summary_file']}")
    
    return 0

if __name__ == "__main__":
    exit(main())
