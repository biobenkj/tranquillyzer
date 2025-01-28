import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

def plot_read_len_distr(parquet_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Parquet files in the directory
    parquet_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) 
                     if f.endswith('.parquet') and f != 'read_index.parquet']
    
    if not parquet_files:
        print("No Parquet files found in the directory.")
        return

    read_lengths = []

    # Iterate through each Parquet file and read the 'ReadLength' column
    for parquet_file in parquet_files:
        try:
            # Read only the 'ReadLength' column
            df = pl.read_parquet(parquet_file, columns=['read_length'])

            # Cast 'ReadLength' column to integers and filter out any null values
            df = df.with_columns(
                pl.col("read_length").cast(pl.Int64, strict=False).alias("read_length")
            ).filter(pl.col("read_length").is_not_null())

            # Collect the 'ReadLength' values into a list
            read_lengths.extend(df["read_length"].to_list())
            
        except Exception as e:
            print(f"Error reading {parquet_file}: {e}")
            continue

    if not read_lengths:
        print("No read lengths found.")
        return

    # Convert read lengths to a NumPy array for processing
    read_lengths = np.array(read_lengths, dtype=int)

    # Check if the read lengths array is not empty and contains expected values
    if len(read_lengths) > 0:
        print(f"Minimum read length: {read_lengths.min()}, Maximum read length: {read_lengths.max()}")
    else:
        print("No valid read lengths found after loading.")
        return

    # Apply log transformation safely (only for positive values)
    log_read_lengths = np.log10(read_lengths[read_lengths > 0])

    # Plot the read length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(log_read_lengths, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Log Read Length Distribution')
    plt.xlabel('Log10(Read Length)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save the plot to the output directory
    plot_file = os.path.join(output_dir, 'read_length_distribution.png')
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Read length distribution plot saved to {plot_file}")

