import pandas as pd
import glob
import os
from pathlib import Path

def combine_csv_files(input_pattern="*.csv", output_file="combined_data.csv"):
    """
    Combine multiple CSV files into a single file, handling missing columns.
    
    Args:
        input_pattern (str): Pattern to match CSV files (e.g., "*.csv" or "combined_log_*.csv")
        output_file (str): Name of the output combined file
    """
    
    # Find all CSV files matching the pattern
    csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Read all CSV files and store them in a list
    dataframes = []
    all_columns = set()
    
    # First pass: collect all unique column names
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_columns.update(df.columns)
            print(f"Read {file}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    # Convert to sorted list for consistent column order
    all_columns = sorted(list(all_columns))
    print(f"\nTotal unique columns found: {len(all_columns)}")
    
    # Second pass: read files and ensure all have the same columns
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # Add missing columns with empty values
            for col in all_columns:
                if col not in df.columns:
                    df[col] = ""  # or use pd.NA or None if you prefer
            
            # Reorder columns to match the master list
            df = df[all_columns]
            
            # Add a source file column for tracking
            df['source_file'] = os.path.basename(file)
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if not dataframes:
        print("No valid dataframes to combine")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save to output file
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCombined data saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")
    
    # Show summary of data from each file
    print(f"\nData summary by source file:")
    source_summary = combined_df.groupby('source_file').size()
    for file, count in source_summary.items():
        print(f"  {file}: {count} rows")

def main():
    """
    Main function to run the CSV combination script.
    You can modify the parameters below as needed.
    """
    
    # Option 1: Combine all CSV files in current directory
    # combine_csv_files("*.csv", "all_combined_data.csv")
    
    # Option 2: Combine specific files matching a pattern
    # combine_csv_files("combined_log_*.csv", "final_combined_data.csv")
    
    # Option 3: Combine specific files by name
    specific_files = ["combined_log_3.csv", "combined_log_4.csv", "combined_log_5.csv"]
    combine_specific_files(specific_files, "combined_log_345.csv")

def combine_specific_files(file_list, output_file="combined_data.csv"):
    """
    Combine specific CSV files by name.
    
    Args:
        file_list (list): List of specific file names to combine
        output_file (str): Name of the output combined file
    """
    
    dataframes = []
    all_columns = set()
    
    # Check if files exist
    existing_files = []
    for file in file_list:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            print(f"Warning: File not found: {file}")
    
    if not existing_files:
        print("No valid files found")
        return
    
    print(f"Processing {len(existing_files)} files:")
    
    # First pass: collect all unique column names
    for file in existing_files:
        try:
            df = pd.read_csv(file)
            all_columns.update(df.columns)
            print(f"  - {file}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    # Convert to sorted list for consistent column order
    all_columns = sorted(list(all_columns))
    print(f"\nTotal unique columns: {len(all_columns)}")
    
    # Second pass: process files
    for file in existing_files:
        try:
            df = pd.read_csv(file)
            
            # Add missing columns with empty values
            for col in all_columns:
                if col not in df.columns:
                    df[col] = ""
            
            # Reorder columns
            df = df[all_columns]
            df['source_file'] = os.path.basename(file)
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        
        print(f"\nCombined data saved to: {output_file}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Total columns: {len(combined_df.columns)}")

if __name__ == "__main__":
    main()
