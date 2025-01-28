import polars as pl
import numpy as np
import pandas as pd
import typer
import os
import csv
import warnings
import random
import logging
import gc
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

import pickle
import tensorflow as tf
import multiprocessing
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from scripts.preprocess_reads import parallel_preprocess_data, find_sequence_files, extract_and_bin_reads, convert_tsv_to_parquet
from scripts.plot_read_len_distr import plot_read_len_distr
from scripts.extract_annotated_seqs import extract_annotated_full_length_seqs
from scripts.annotate_new_data import annotate_new_data
from scripts.visualize_annot import save_plots_to_pdf
from scripts.export_annotations import process_full_length_reads_in_chunks_and_save, load_checkpoint, save_checkpoint
from scripts.trained_models import trained_models, seq_orders
from scripts.correct_barcodes import generate_barcodes_stats_pdf
from scripts.demultiplex import generate_demux_stats_pdf

app = typer.Typer()

############# available trained models ################

@app.command()
def availableModels():
    trained_models()

############# extract reads, read_names from fasta file ################

@app.command()
def preprocessFasta(fasta_dir: str, output_dir: str,
                    threads: int = typer.Argument(1),
                    portion: str = typer.Argument("end"), 
                    end_length: int = typer.Argument(250),
                    batch_size: int = typer.Argument(100000)):
    
    os.system("mkdir -p " + output_dir + "/full_length_pp_fa")
    if portion == "end":
        os.system("mkdir -p " + output_dir + "/read_ends_pp_fa")

    # parallel_preprocess_data(fasta_dir, output_dir, portion, end_length, batch_size, threads)
    files_to_process = find_sequence_files(fasta_dir)
    if len(files_to_process) == 1:
        # If there is only one file, process it in a single thread
        logger.info("Only one file to process. Processing without parallelization.")
        extract_and_bin_reads(files_to_process[0], batch_size, output_dir + "/full_length_pp_fa")
        
        os.system(f"rm {output_dir}/full_length_pp_fa/*.lock")
        convert_tsv_to_parquet(f"{output_dir}/full_length_pp_fa", row_group_size=1000000)
    else:
        # Process files in parallel
        parallel_preprocess_data(files_to_process, output_dir + "/full_length_pp_fa", batch_size, num_workers=threads)
    # convert_tsv_to_parquet(output_dir + "/full_length_pp_fa", chunk_size=1000000)
    
# ############# plot read length distribution ################

@app.command()
def readlengthDist(output_dir: str):
    os.system("mkdir -p " + output_dir + "/plots")
    plot_read_len_distr(output_dir + "/full_length_pp_fa", output_dir + "/plots")

############# inspect selected reads for annotations ################

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_read_index(index_file_path, read_name):
    """Load the specific row from the read_index.parquet for a given read name."""
    df = pl.read_parquet(index_file_path).filter(pl.col("ReadName") == read_name)
    if df.is_empty():
        logger.warning(f"Read {read_name} not found in the index.")
        return None
    return df["ParquetFile"][0]  # Return the associated Parquet file for the read name

@app.command()
def visualize(model_name: str, 
              output_dir: str,
              num_reads: int = typer.Option(None, help="Number of reads to randomly visualize from each Parquet file."),
              portion: str = typer.Option("full", help="Whether to scan full-reads or just the ends"),
              end_length: int = typer.Option(250, help="How many bases to be scanned from the ends"),
              read_names: str = typer.Option(None, help="Comma-separated list of read names to visualize")):
    
    model = "models/" + model_name + ".h5"
    with open("models/" + model_name + "_lbl_bin.pkl", "rb") as f:
        label_binarizer = pickle.load(f)

    seq_order, sequences, barcodes, UMIs = seq_orders("utils/seq_orders.tsv", model_name)

    palette = ['red', 'blue', 'green', 'purple', 'pink', 'cyan', 'magenta', 'orange', 'brown']
    colors = {'random_s': 'black', 'random_e': 'black', 'cDNA': 'gray', 'polyT': 'orange', 'polyA': 'orange'}

    i = 0
    for element in seq_order:
        if element not in ['random_s', 'random_e', 'cDNA', 'polyT', 'polyA']:
            colors[element] = palette[i % len(palette)]  # Cycle through the palette
            i += 1

    # Path to the read_index.parquet
    index_file_path = os.path.join(output_dir, "full_length_pp_fa/read_index.parquet")

    if portion == "full":
        folder_path = os.path.join(output_dir, "full_length_pp_fa")
        pdf_filename = os.path.join(output_dir, "plots/full_read_annots.pdf")
    else:
        folder_path = os.path.join(output_dir, "read_ends_pp_fa")
        pdf_filename = os.path.join(output_dir, "plots/read_ends_annots.pdf")

    if not num_reads and not read_names:
        logger.error("You must either provide a value for 'num_reads' or specify 'read_names'.")
        raise ValueError("You must either provide a value for 'num_reads' or specify 'read_names'.")

    selected_reads = []
    selected_read_names = []
    selected_read_lengths = []

    # If read_names are provided, visualize those specific reads
    if read_names:
        read_names_list = read_names.split(",")
        missing_reads = []

        for read_name in read_names_list:
            parquet_file = load_read_index(index_file_path, read_name)

            if parquet_file:
                parquet_path = os.path.join(folder_path, parquet_file)

                try:
                    # Load the appropriate Parquet file and retrieve the read
                    df = pl.read_parquet(parquet_path).filter(pl.col("ReadName") == read_name)
                    if not df.is_empty():
                        read_seq = df["read"][0]
                        read_length = df["read_length"][0]
                        selected_reads.append(read_seq)
                        selected_read_names.append(read_name)
                        selected_read_lengths.append(read_length)
                except Exception as e:
                    logger.error(f"Error reading {parquet_path}: {e}")
            else:
                missing_reads.append(read_name)

        if missing_reads:
            logger.warning(f"The following reads were not found in the index: {', '.join(missing_reads)}")

    # If num_reads is provided, randomly select num_reads reads from the index
    elif num_reads:
        df_index = pl.read_parquet(index_file_path)
        all_read_names = df_index["ReadName"].to_list()
        selected_read_names = random.sample(all_read_names, min(num_reads, len(all_read_names)))

        for read_name in selected_read_names:
            parquet_file = load_read_index(index_file_path, read_name)

            if parquet_file:
                parquet_path = os.path.join(folder_path, parquet_file)

                try:
                    df = pl.read_parquet(parquet_path).filter(pl.col("ReadName") == read_name)
                    if not df.is_empty():
                        read_seq = df["Read"][0]
                        read_length = df["ReadLength"][0]
                        selected_reads.append(read_seq)
                        selected_read_lengths.append(read_length)
                except Exception as e:
                    logger.error(f"Error reading {parquet_path}: {e}")

    # Check if there are any selected reads to process
    if not selected_reads:
        logger.warning("No reads were selected. Skipping inference.")
        return

    # Perform annotation and plotting
    if portion == "full":
        predictions = annotate_new_data(selected_reads, model)
        annotated_reads = extract_annotated_full_length_seqs(
            selected_reads, predictions, selected_read_lengths, label_binarizer, seq_order, barcodes, n_jobs=1
        )
        save_plots_to_pdf(selected_reads, annotated_reads, selected_read_names, pdf_filename, colors, chars_per_line=150)
    
############# Annotate all the reads ################

# Function to calculate the total number of rows in the Parquet file
def calculate_total_rows(parquet_file):
    df = pl.scan_parquet(parquet_file)
    total_rows = df.collect().shape[0]
    return total_rows

# Modified function to estimate the average read length from the bin name
def estimate_average_read_length_from_bin(bin_name):
    bounds = bin_name.replace("bp", "").split("_")
    lower_bound = int(bounds[0])
    upper_bound = int(bounds[1])
    return (lower_bound + upper_bound) / 2

# Function to process the Parquet file in chunks with dynamic chunk sizing
def load_and_process_reads_by_bin(parquet_file, chunk_start, chunk_size, model, 
                                  cumulative_barcodes_stats, reason_counter,
                                  label_binarizer, all_read_lengths, all_cDNA_lengths,
                                  match_type_counter, cell_id_counter, 
                                  seq_order, output_dir, add_header, checkpoint_file, 
                                  barcodes, whitelist_df, n_jobs):
    
    total_rows = calculate_total_rows(parquet_file)
    bin_name = os.path.basename(parquet_file).replace(".parquet", "")
    
    # Estimate the average read length from the bin name and adjust chunk size
    estimated_avg_length = estimate_average_read_length_from_bin(bin_name)
    dynamic_chunk_size = int(chunk_size * (500 / estimated_avg_length))  # Scale chunk size dynamically

    # Read the input file in chunks

    num_chunks = (total_rows // dynamic_chunk_size) + (1 if total_rows % dynamic_chunk_size > 0 else 0)

    # Iterate over chunks within the Parquet file
    for chunk_idx in range(chunk_start, num_chunks + 1):
        print(f"Processing {bin_name}: chunk {chunk_idx}")
        
        # Read the current chunk of rows from the Parquet file
        df_chunk = pl.scan_parquet(parquet_file).slice((chunk_idx - 1) * dynamic_chunk_size, dynamic_chunk_size).collect()
        read_names = df_chunk["ReadName"].to_list()
        reads = df_chunk["read"].to_list()
        read_lengths = df_chunk["read_length"].to_list()
        
        # output_file = f"{output_path}/{bin_name}.tsv"

        if chunk_idx == 1:
            append = "w"
        else:
            append = "a"

        # Process the current chunk (full-length reads)
        results = process_full_length_reads_in_chunks_and_save(reads, read_names, model, label_binarizer, cumulative_barcodes_stats,
                                                               reason_counter, read_lengths, seq_order, add_header, append, bin_name,
                                                               output_dir, barcodes, whitelist_df, n_jobs)
        
        if results is not None:
            match_type_counts, cell_id_counts, cDNA_lengths, cumulative_barcodes_stats, reason_counter = results

            # Update cumulative stats;k
            all_read_lengths.extend(read_lengths)
            all_cDNA_lengths.extend(cDNA_lengths)
            
            for key, value in match_type_counts.items():
                match_type_counter[key] += value
            for key, value in cell_id_counts.items():
                cell_id_counter[key] += value
            
        save_checkpoint(checkpoint_file, bin_name, chunk_start)
            
        add_header = False  # Only add header for the first chunk
        gc.collect()  # Clean up memory after processing each chunk

    return cumulative_barcodes_stats, all_read_lengths, all_cDNA_lengths, match_type_counter, cell_id_counter, reason_counter

def filtering_reason_stats(reason_counter_by_bin, output_dir):
    """Plot a heatmap with a bar chart while improving y-axis readability.
    Saves both raw count and normalized fraction TSVs.
    """

    # Convert dictionary to DataFrame (Bins as Columns, Reasons as Rows)
    raw_counts_df = pd.DataFrame.from_dict(reason_counter_by_bin, orient='index').fillna(0).T

    # Get correct y-axis order from the dictionary
    correct_reason_order = list(raw_counts_df.index)

    # Compute total reads per bin
    total_reads = raw_counts_df.sum(axis=0)

    # Normalize each column (fraction per bin)
    normalized_data = raw_counts_df.div(total_reads, axis=1)

    # Save both raw counts and normalized fractions
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    raw_counts_df.to_csv(f"{output_dir}/filtered_raw_counts_by_bins.tsv", sep='\t')
    normalized_data.to_csv(f"{output_dir}/filtered_normalized_fractions_by_bins.tsv", sep='\t')

    print(f"Saved raw counts to {output_dir}/filtered_raw_counts_by_bins.tsv")
    print(f"Saved normalized fractions to {output_dir}/filtered_normalized_fractions_by_bins.tsv")

def plot_reason_heatmap_from_tsv(output_dir):
    """Replot heatmap with log10 bar chart using saved TSV data."""

    raw_counts_file = f"{output_dir}/filtered_raw_counts_by_bins.tsv"
    normalized_fractions_file = f"{output_dir}/filtered_normalized_fractions_by_bins.tsv"

    # Load the normalized fractions
    normalized_data = pd.read_csv(normalized_fractions_file, sep='\t', index_col=0)

    # Check if raw counts are available for bar chart
    if os.path.exists(raw_counts_file):
        raw_counts = pd.read_csv(raw_counts_file, sep='\t', index_col=0)
        total_reads = raw_counts.sum(axis=0)
        log_total_reads = np.log10(total_reads.replace(0, np.nan))
    else:
        log_total_reads = None  # Can't generate bar chart

    # Create figure with gridspec layout
    fig = plt.figure(figsize=(14, 30))
    gs = GridSpec(2, 1, height_ratios=[1, 5])

    # Plot bar chart if we have total reads
    if log_total_reads is not None:
        ax0 = plt.subplot(gs[0])
        ax0.bar(normalized_data.columns, log_total_reads, color='gray')
        ax0.set_title("Total Reads per Bin (log10 scale)", fontsize=14, pad=15)
        ax0.set_ylabel("log10(Total Reads)", fontsize=12)
        ax0.set_xticklabels([])
    else:
        print("Raw counts not found, skipping bar chart.")

    # Plot heatmap
    ax1 = plt.subplot(gs[1])
    sns.heatmap(
        normalized_data,
        cmap="YlGnBu",
        annot=False,
        cbar_kws={'label': 'Fraction of Reads'},
        ax=ax1
    )

    # Set labels and formatting
    ax1.set_title("Reason for filtering vs read length", fontsize=14, pad=15)
    ax1.set_xlabel("Bin Name", fontsize=12, labelpad=10)
    ax1.set_ylabel("Reason", fontsize=12, labelpad=10)
    ax1.set_xticklabels(normalized_data.columns, rotation=45, ha='right')

    # Adjust layout
    plt.subplots_adjust(left=0.4, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.3)

    # Save figure
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    plt.savefig(f"{output_dir}/plots/filtering_reason_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved heatmap to {output_dir}/plots/filtering_reason_heatmap.png")

# Modified function to process the entire directory of Parquet files in chunks with dynamic chunk sizing
@app.command()
def annotate_reads(
    model_name: str, 
    output_dir: str, 
    whitelist_file: str,
    chunk_size: int = typer.Option(100000, help="Base chunk size, will adjust dynamically based on read length"),
    portion: str = typer.Option("end", help="Whether to process the entire reads or just the ends"), 
    end_length: int = typer.Option(250, help="Bases from each end to be processed"),
    njobs: int = typer.Option(12, help="number of CPU threads for barcode correction and demultiplexing")):
    
    # Model and label binarizer loading
    model = "models/" + model_name + ".h5"
    
    with open("models/" + model_name + "_lbl_bin.pkl", "rb") as f:
        label_binarizer = pickle.load(f)

    # Load sequence order
    seq_order, sequences, barcodes, UMIs = seq_orders("utils/seq_orders.tsv", model_name)

    whitelist_df = pd.read_csv(whitelist_file, sep='\t')

    # Set base folder path depending on whether we are processing full reads or ends
    base_folder_path = os.path.join(output_dir, "full_length_pp_fa") if portion == "full" else os.path.join(output_dir, "read_ends_pp_fa")

    # # Get the list of all Parquet files (excluding read_index.parquet)
    parquet_files = [os.path.join(base_folder_path, f) for f in os.listdir(base_folder_path) 
                     if f.endswith('.parquet') and not f.endswith('read_index.parquet')]

    # Sort Parquet files by estimated average read length (shorter first)
    parquet_files.sort(key=lambda f: estimate_average_read_length_from_bin(os.path.basename(f).replace(".parquet", "")))

    count = 0

    column_mapping = {}

    for barcode in barcodes:
        column_mapping[barcode] = barcode

    cumulative_barcodes_stats = {barcode: {'count_data': {}, 'min_dist_data': {}} for barcode in list(column_mapping.keys())}

    all_read_lengths = []
    all_cDNA_lengths = []

    match_type_counter = defaultdict(int)
    cell_id_counter = defaultdict(int)
    reason_counter_by_bin = {}

    # Process each Parquet file, sorted by read length
    for parquet_file in parquet_files:
        reason_counter = defaultdict(int)
        bin_name = os.path.basename(parquet_file).replace(".parquet", "")
        os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)

        # Load checkpoint if available
        checkpoint_file = os.path.join(os.path.join(output_dir, "tmp"), 
                                       "annotation_checkpoint.txt")
        last_bin, last_chunk = load_checkpoint(checkpoint_file, bin_name)
        
        # If we're starting a new bin, reset chunk_start
        chunk_start = last_chunk if last_bin == bin_name else 1

        add_header = True if count == 0 and chunk_start == 1 else False

        result = load_and_process_reads_by_bin(parquet_file, chunk_start, chunk_size, model, 
                                                cumulative_barcodes_stats, reason_counter,
                                                label_binarizer, all_read_lengths, all_cDNA_lengths,
                                                match_type_counter, cell_id_counter,
                                                seq_order, output_dir, add_header, checkpoint_file,
                                                barcodes, whitelist_df, njobs)
        if result is not None:
            cumulative_barcodes_stats, all_read_lengths, all_cDNA_lengths, match_type_counter, cell_id_counter, reason_counter = result
        
        reason_counter_by_bin[bin_name] = reason_counter
        count += 1
        gc.collect()  # Clean up memory after each file is processed

    generate_barcodes_stats_pdf(cumulative_barcodes_stats, list(column_mapping.keys()), 
                                pdf_filename=output_dir + "/plots/barcode_plots.pdf")
    generate_demux_stats_pdf(output_dir + "/plots/demux_plots.pdf", match_type_counter,
                             cell_id_counter, all_read_lengths, all_cDNA_lengths)
    # plot_reason_heatmap_with_bar(reason_counter_by_bin, output_dir)
    filtering_reason_stats(reason_counter_by_bin, output_dir)

    plot_reason_heatmap_from_tsv(output_dir)

    # Concatenate valid results into a single output file
    output_path = os.path.join(output_dir, "tmp")

    tsv_files_valid = sorted(glob.glob(os.path.join(output_path, "*_valid.tsv")))
    
    if not tsv_files_valid:
        logger.info("No valid .tsv files found to combine.")
    else:
        logger.info(f"Found {len(tsv_files_valid)} valid .tsv files. Combining into annotations_valid.tsv")
        with open(output_dir + "/annotations_valid.tsv", 'w') as outfile:
            for tsv_file in tsv_files_valid:
                with open(tsv_file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)  # Write each line to the output file
                    infile.close() 
            outfile.close()
        logger.info("Merging complete.")

    df = pl.scan_csv(f"{output_dir}/annotations_valid.tsv", separator='\t')
    annotations_valid_parquet_file = f"{output_dir}/annotations_valid.parquet"
    
    logger.info(f"Converting annotations_valid.tsv")
    df.sink_parquet(annotations_valid_parquet_file, compression="snappy", row_group_size=chunk_size)
    logger.info(f"Converted annotations_valid.tsv to annotations_valid.parquet")

    os.system(f"rm {output_dir}/annotations_valid.tsv")

    # Concatenate invalid results into a single output file

    tsv_files_invalid = sorted(glob.glob(os.path.join(output_path, "*_invalid.tsv")))
    
    if not tsv_files_invalid:
        logger.info("No invalid .tsv files found to combine.")
    else:
        logger.info(f"Found {len(tsv_files_invalid)} invalid .tsv files. Combining into annotations_invalid.tsv")
        with open(output_dir + "/annotations_invalid.tsv", 'w') as outfile:
            for tsv_file in tsv_files_invalid:
                with open(tsv_file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)  # Write each line to the output file
                    infile.close() 
            outfile.close()
        logger.info("Merging complete.")

    df = pl.scan_csv(f"{output_dir}/annotations_invalid.tsv", separator='\t',
                     dtypes={"UMI_Starts": pl.Utf8,  "UMI_Ends": pl.Utf8})
    annotations_invalid_parquet_file = f"{output_dir}/annotations_invalid.parquet"
    
    logger.info(f"Converting annotations_invalid.tsv")
    df.sink_parquet(annotations_invalid_parquet_file, compression="snappy", row_group_size=chunk_size)
    logger.info(f"Converted annotations_invalid.tsv to annotations_invalid.parquet")

    os.system(f"rm {output_dir}/annotations_invalid.tsv")

    os.system(f"rm -r {output_path}")

if __name__ == "__main__":
    app()