import polars as pl
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
import pandas as pd
from collections import defaultdict

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from scripts.preprocess_reads import parallel_preprocess_data, find_sequence_files, extract_and_bin_reads, convert_tsv_to_parquet
from scripts.plot_read_len_distr import plot_read_len_distr
from scripts.extract_annotated_seqs import extract_annotated_full_length_seqs, extract_annotated_seq_ends
from scripts.annotate_new_data import annotate_new_data
from scripts.visualize_annot import save_plots_to_pdf
from scripts.export_annotations import process_read_ends_in_chunks_and_save, process_full_length_reads_in_chunks_and_save, load_checkpoint, save_checkpoint
from scripts.simulate_training_data import prepare_training_data
from scripts.train_model import train_model, ReadAnnotator
from scripts.trained_models import trained_models, seq_orders
from scripts.correct_UMI import process_and_correct_umis
from scripts.correct_barcodes import barcode_correction_pipeline
from scripts.demultiplex import assign_cell_id_in_chunks
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

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def load_read_index(index_file_path):
#     """Load the read index from the read_index.tsv file."""
#     read_index = {}
#     with open(index_file_path, 'r') as file:
#         reader = csv.reader(file, delimiter='\t')
#         next(reader)  # Skip header
#         for row in reader:
#             read_name, pkl_file, read_idx, read_length = row
#             read_index[read_name] = (pkl_file, int(read_idx))
#     return read_index

# @app.command()
# def visualize(model_name: str, 
#               output_dir: str,
#               num_reads: int = typer.Option(None, help="Number of reads to randomly visualize from each .pkl file."),
#               portion: str = typer.Option("full", help="Whether to scan full-reads or just the ends"),
#               end_length: int = typer.Option(250, help="How many bases to be scanned from the ends"),
#               read_names: str = typer.Option(None, help="Comma-separated list of read names to visualize")):
    
#     # model = load_model("models/" + model_name + ".h5")
#     model = "models/" + model_name + ".h5"
#     with open("models/" + model_name + "_lbl_bin.pkl", "rb") as f:
#         label_binarizer = pickle.load(f)

#     seq_order, sequences, barcodes, UMIs = seq_orders("utils/seq_orders.tsv", model_name)

#     palette = ['red', 'blue', 'green', 'purple', 'pink', 'cyan', 'magenta', 'orange', 'brown']
#     colors = {'random_s': 'black', 'random_e': 'black', 'cDNA': 'gray', 'polyT': 'orange', 'polyA': 'orange'}

#     i = 0
#     for element in seq_order:
#         if element not in ['random_s', 'random_e', 'cDNA', 'polyT', 'polyA']:
#             colors[element] = palette[i % len(palette)]  # Cycle through the palette
#             i += 1

#     # Load the read index from read_index.tsv
#     index_file_path = os.path.join(output_dir, "full_length_pp_fa/read_index.tsv")
#     read_index = load_read_index(index_file_path)

#     if portion == "full":
#         folder_path = os.path.join(output_dir, "full_length_pp_fa")
#         pdf_filename = os.path.join(output_dir, "full_read_annots.pdf")
#     else:
#         folder_path = os.path.join(output_dir, "read_ends_pp_fa")
#         pdf_filename = os.path.join(output_dir, "read_ends_annots.pdf")

#     # Check conditions for num_reads and read_names
#     if not num_reads and not read_names:
#         logger.error("You must either provide a value for 'num_reads' or specify 'read_names'.")
#         raise ValueError("You must either provide a value for 'num_reads' or specify 'read_names'.")

#     selected_reads = []
#     selected_read_names = []
#     selected_read_lengths = []

#     # If read_names are provided, visualize those specific reads
#     if read_names:
#         read_names_list = read_names.split(",")
#         missing_reads = []

#         for read_name in read_names_list:
#             if read_name in read_index:
#                 pkl_file, read_idx = read_index[read_name]

#                 # Load the appropriate pkl file and retrieve the read by index
#                 with open(os.path.join(folder_path, pkl_file), 'rb') as f:
#                     reads_data = pickle.load(f)
#                     selected_reads.append(reads_data['reads'][read_idx])
#                     selected_read_names.append(read_name)
#                     selected_read_lengths.append(reads_data['read_lengths'][read_idx])
#             else:
#                 missing_reads.append(read_name)

#         # Log missing reads
#         if missing_reads:
#             logger.warning(f"The following reads were not found in the index: {', '.join(missing_reads)}")

#     # If num_reads is provided, randomly select num_reads reads from the index
#     elif num_reads:
#         all_read_names = list(read_index.keys())
#         selected_read_names = random.sample(all_read_names, min(num_reads, len(all_read_names)))

#         for read_name in selected_read_names:
#             pkl_file, read_idx = read_index[read_name]

#             # Load the appropriate pkl file and retrieve the read by index
#             with open(os.path.join(folder_path, pkl_file), 'rb') as f:
#                 reads_data = pickle.load(f)
#                 selected_reads.append(reads_data['reads'][read_idx])
#                 selected_read_lengths.append(reads_data['read_lengths'][read_idx])

#     # Check if there are any selected reads to process
#     if not selected_reads:
#         logger.warning(f"No reads were selected. Skipping inference.")
#         return  # Exit if no reads are selected

#     # Perform annotation and plotting
#     if portion == "full":
#         predictions = annotate_new_data(selected_reads, model)
#         annotated_reads = extract_annotated_full_length_seqs(selected_reads, predictions, selected_read_lengths, 
#                                                              label_binarizer, seq_order, n_jobs=1)
#         save_plots_to_pdf(selected_reads, annotated_reads, selected_read_names, 
#                           pdf_filename, colors, chars_per_line=150)
#     else:
#         read_ends = random.sample(reads_data['read_ends'], min(num_reads * 2, len(reads_data['read_ends'])))
#         indices = [reads_data['read_ends'].index(read) for read in read_ends]
#         read_end_names = [reads_data['read_end_names'][i] for i in indices]
#         actual_lengths = [reads_data['actual_lengths'][i] for i in indices]

#         predictions = annotate_new_data(read_ends, model)
#         annotated_reads = extract_annotated_seq_ends(read_ends, predictions, label_binarizer, actual_lengths, end_length, seq_order)
#         save_plots_to_pdf(read_ends, annotated_reads, read_end_names, pdf_filename, colors, chars_per_line=150)

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
    else:
        read_ends = [read[:end_length] for read in selected_reads] + [read[-end_length:] for read in selected_reads]
        predictions = annotate_new_data(read_ends, model)
        annotated_reads = extract_annotated_seq_ends(
            read_ends, predictions, label_binarizer, selected_read_lengths, end_length, seq_order
        )
        save_plots_to_pdf(read_ends, annotated_reads, selected_read_names, pdf_filename, colors, chars_per_line=150)

############# Annotate all the reads ################

# # Modified function to estimate the average read length from the bin name
# def estimate_average_read_length_from_bin(bin_name):
#     bounds = bin_name.replace("bp", "").split("_")
#     lower_bound = int(bounds[0])
#     upper_bound = int(bounds[1])
#     return (lower_bound + upper_bound) / 2

# # Function to process reads in chunks and resume from where it left off
# def load_and_process_reads_by_bin(bin_name, chunk_start, folder, chunk_size, model, label_binarizer, seq_order, 
#                                   output_file, add_header, checkpoint_file):
    
#     estimated_avg_length = estimate_average_read_length_from_bin(bin_name)
#     bin_file_path = os.path.join(folder, f"{bin_name}.pkl")

#     # Check if the bin file exists in this folder
#     if not os.path.exists(bin_file_path):
#         print(f"Skipping folder {folder} as it doesn't contain {bin_name}.pkl")
#         return
            
#     # Load the reads from the .pkl file
#     with open(bin_file_path, "rb") as f:
#         reads_data = pickle.load(f)
#         reads = reads_data['reads']
#         read_names = reads_data['read_names']
#         read_lengths = reads_data['read_lengths']
            
#     dynamic_chunk_size = int(chunk_size * (500 / estimated_avg_length))  # Scale chunk size
            
#     # Start processing from the last chunk if available, or from the beginning
#     process_full_length_reads_in_chunks_and_save(reads, read_names, model, label_binarizer, 
#                                                  read_lengths, seq_order, dynamic_chunk_size, 
#                                                  bin_name, add_header, output_path=output_file, 
#                                                  chunk_start=chunk_start, checkpoint_file=checkpoint_file)                  

# @app.command()
# def annotate_reads(
#     model_name: str, 
#     output_dir: str, 
#     chunk_size: int = typer.Option(100000, help="Base chunk size, will adjust dynamically based on read length"),
#     portion: str = typer.Option("end", help="Whether to process the entire reads or just the ends"), 
#     end_length: int = typer.Option(250, help="Bases from each end to be processed")):
    
#     # Model and label binarizer loading
#     model = "models/" + model_name + ".h5"
#     # model = load_model(model)

#     with open("models/" + model_name + "_lbl_bin.pkl", "rb") as f:
#         label_binarizer = pickle.load(f)

#     # Load sequence order
#     seq_order, sequences, barcodes, UMIs = seq_orders("utils/seq_orders.tsv", model_name)
    
#     # seq_order.insert(0, "cDNA")
#     # seq_order.append("cDNA")

#     # Determine folder path based on whether we're processing full-length reads or read ends
#     if portion == "full":
#         base_folder_path = os.path.join(output_dir, "full_length_pp_fa")
#     else:
#         base_folder_path = os.path.join(output_dir, "read_ends_pp_fa")
    
#     # Get the list of subdirectories (e.g., file1, file2, etc.)
#     folder_paths = [os.path.join(base_folder_path, folder) for folder in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, folder))]

#     # Get the bin names dynamically by listing all .pkl files and using the read length estimates from bin names
    
#     count = 0

#     for folder in folder_paths:
#         bin_names = []
        
#         output_path = output_dir + "/tmp/" + os.path.basename((folder)) + "_annotation_tmp"
#         os.system("cd " + output_dir + "\nmkdir -p tmp/" + os.path.basename((folder)) + "_annotation_tmp")

#         for file in os.listdir(folder):
#             if file.endswith(".pkl"):
#                 bin_name = file.replace(".pkl", "")
#                 bin_names.append(bin_name)
        
#         # Sort bins by estimated average read length (shorter first)
#         bin_names.sort(key=lambda bin_name: estimate_average_read_length_from_bin(bin_name))

#         checkpoint_file = output_path + "/annotation_checkpoint.txt"

#         # Load checkpoint if available
#         last_bin, last_chunk = load_checkpoint(checkpoint_file, bin_names[0])
#         chunk_start = last_chunk

#         # Process each bin dynamically found in the folder, sorted by read length
#         for i in range(bin_names.index(last_bin), len(bin_names)):
#             bin_name = bin_names[i]
#             if last_bin == bin_names[i-1]:
#                 chunk_start = 1

#             if bin_name == bin_names[0] and chunk_start == 1 and count == 0:
#                 add_header = True
#             else:
#                 add_header = False
#             load_and_process_reads_by_bin(bin_name, chunk_start, folder, chunk_size, model, 
#                                           label_binarizer, seq_order, output_path, add_header,
#                                           checkpoint_file)

#         if count == 0:
#             os.system("cat " + output_path + "/*.tsv > " + output_dir + "/annotations.tsv")
#         else:
#             os.system("cat " + output_path + "/*.tsv >> " + output_dir + "/annotations.tsv")
        
#         count += 1

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
                                  cumulative_barcodes_stats, label_binarizer, 
                                  all_read_lengths, all_cDNA_lengths,
                                  match_type_counter, cell_id_counter, 
                                  seq_order, output_dir, add_header, checkpoint_file, 
                                  barcodes, whitelist_df, n_jobs):
    total_rows = calculate_total_rows(parquet_file)
    bin_name = os.path.basename(parquet_file).replace(".parquet", "")
    
    # Estimate the average read length from the bin name and adjust chunk size
    estimated_avg_length = estimate_average_read_length_from_bin(bin_name)
    dynamic_chunk_size = int(chunk_size * (500 / estimated_avg_length))  # Scale chunk size dynamically

    # Read the input file in chunks

    # Iterate over chunks within the Parquet file
    for chunk_idx in range(chunk_start, (total_rows // dynamic_chunk_size) + 1):
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
                                                               read_lengths, seq_order, add_header, append, bin_name, output_dir,
                                                               barcodes, whitelist_df, n_jobs)
        
        if results is not None:
            match_type_counts, cell_id_counts, cDNA_lengths, cumulative_barcodes_stats = results

            # Update cumulative stats
            all_read_lengths.extend(read_lengths)
            all_cDNA_lengths.extend(cDNA_lengths)
            
            for key, value in match_type_counts.items():
                match_type_counter[key] += value
            for key, value in cell_id_counts.items():
                cell_id_counter[key] += value
            
        save_checkpoint(checkpoint_file, bin_name, chunk_start)
            
        add_header = False  # Only add header for the first chunk
        gc.collect()  # Clean up memory after processing each chunk

    return cumulative_barcodes_stats, all_read_lengths, all_cDNA_lengths, match_type_counter, cell_id_counter

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

    # Get the list of all Parquet files (excluding read_index.parquet)
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

    # Process each Parquet file, sorted by read length
    for parquet_file in parquet_files:
        bin_name = os.path.basename(parquet_file).replace(".parquet", "")
        # output_path = os.path.join(output_dir, "tmp")
        os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)

        # Load checkpoint if available
        checkpoint_file = os.path.join(os.path.join(output_dir, "tmp"), 
                                       "annotation_checkpoint.txt")
        last_bin, last_chunk = load_checkpoint(checkpoint_file, bin_name)
        
        # If we're starting a new bin, reset chunk_start
        chunk_start = last_chunk if last_bin == bin_name else 1

        add_header = True if count == 0 and chunk_start == 1 else False

        result = load_and_process_reads_by_bin(parquet_file, chunk_start, chunk_size, model, 
                                                cumulative_barcodes_stats, label_binarizer, 
                                                all_read_lengths, all_cDNA_lengths,
                                                match_type_counter, cell_id_counter,
                                                seq_order, output_dir, add_header, checkpoint_file,
                                                barcodes, whitelist_df, njobs)
        if result is not None:
            cumulative_barcodes_stats, all_read_lengths, all_cDNA_lengths, match_type_counter, cell_id_counter = result
        
        count += 1
        gc.collect()  # Clean up memory after each file is processed

    generate_barcodes_stats_pdf(cumulative_barcodes_stats, list(column_mapping.keys()), 
                                pdf_filename=output_dir + "/plots/barcode_plots.pdf")
    generate_demux_stats_pdf(output_dir + "/plots/demux_plots.pdf", match_type_counter,
                             cell_id_counter, all_read_lengths, all_cDNA_lengths)

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

    df = pl.scan_csv(f"{output_dir}/annotations_invalid.tsv", separator='\t')
    annotations_invalid_parquet_file = f"{output_dir}/annotations_invalid.parquet"
    
    logger.info(f"Converting annotations_invalid.tsv")
    df.sink_parquet(annotations_invalid_parquet_file, compression="snappy", row_group_size=chunk_size)
    logger.info(f"Converted annotations_invalid.tsv to annotations_invalid.parquet")

    os.system(f"rm {output_dir}/annotations_invalid.tsv")

    os.system(f"rm -r {output_path}")

############# correct barcodes ############

@app.command()
def correct_barcodes(model_name: str,
                     annotation_file: str,
                     whitelist_file: str,
                     output_dir: str, 
                     threshold: int = typer.Option(2, help="Levenshtein distance threshold"),
                     threads: int = typer.Option(2, help="Number of threads"),
                     chunksize: int = typer.Option(100000, help="Chunk of reads to be processed at a time")):
    
    seq_order, sequences, barcodes, UMIs = seq_orders("utils/seq_orders.tsv", model_name)

    os.system("grep -v invalid " + annotation_file + " | grep -v , > " + output_dir + "/valid_annotations.tsv")
                                                                                                                                     
    # output_file = output_dir + "/corrected_barcodes.tsv"

    column_mapping = {}

    for barcode in barcodes:
        column_mapping[barcode] = barcode

    barcode_correction_pipeline(output_dir + "/valid_annotations.tsv", 
                                whitelist_file, output_dir, column_mapping, 
                                threshold, threads, chunksize)

############# demultiplex ############

@app.command()
def demultiplex(whitelist_file: str,
                output_dir: str,
                chunk_size: int = typer.Option(1000000, help="Number of reads to be processes in a batch"),
                threads: int = typer.Option(1, help="Number of threads")):
    
    # seq_order, sequences, barcodes, UMIs = seq_orders("utils/seq_orders.tsv", model_name)

    # column_mapping = {}

    # for barcode in barcodes:
    #     column_mapping[barcode] = barcode

    corrected_barcodes_file = output_dir + "/corrected_barcodes.tsv"

    assign_cell_id_in_chunks(corrected_barcodes_file, whitelist_file, output_dir, chunk_size, num_cores=threads)

    print("Demultiplexing complete. Output saved to demultiplexed_reads.tsv")

############# Correct UMI ################

@app.command()
def correct_UMI(output_dir: str, 
                chunk_size: int = typer.Option(10000, help="Number of reads to be processes in a batch"),
                threads: int = typer.Option(2, help="Number of threads"),
                umi_column: str = typer.Option("UMI_Sequences", help="Name of the UMI column in the annotations.tsv file")):

    input_file = output_dir + "/end_annotations_filt.tsv"
    output_file = output_dir + "/umi_corrected.tsv" 

    process_and_correct_umis(input_file, output_file, umi_column=umi_column, threshold=2, num_workers=threads)
    # process_and_correct_umis(input_file, output_file, umi_column='UMI_Sequences', threshold=2)

############ Simulate training data ##############

@app.command()
def simulate_training_data(model_name: str, output_dir: str,
                           read_type: str = typer.Argument("single"), 
                           threads: int = typer.Argument(1),
                           mismatch_rate: float = typer.Argument(0.044),
                           insertion_rate: float = typer.Argument(0.066116500187765),
                           deletion_rate: float = typer.Argument(0.0612981959469103)):
    
    seq_order, sequences, barcodes, UMIs = seq_orders("utils/seq_orders.tsv", model_name)

    X_train, X_val, Y_train, Y_val, label_binarizer, max_seq_len = prepare_training_data(seq_order, sequences, read_type, threads,
                                                                                         mismatch_rate, insertion_rate, deletion_rate)

    with open(output_dir + "/X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(output_dir + "/X_val.pkl", "wb") as f:
        pickle.dump(X_val, f)
    with open(output_dir + "/Y_train.pkl", "wb") as f:
        pickle.dump(Y_train, f)
    with open(output_dir + "/Y_val.pkl", "wb") as f:
        pickle.dump(Y_val, f)
        
    with open(output_dir + "/label_binarizer.pkl", "wb") as f:
        pickle.dump(label_binarizer, f)
    with open(output_dir + "/max_seq_len.pkl", "wb") as f:
         pickle.dump(max_seq_len, f)


############ Train a model #############

@app.command()
def train_new_model(data_dir: str, model_dir: str, 
                    embedding_dim: int, num_labels: int,
                    num_conv_layers: int = typer.Argument(4),
                    num_lstm_layers: int = typer.Argument(4),
                    epochs: int = typer.Argument(5),
                    batch_size: int = typer.Argument(150)):
    
    with open(data_dir + "/X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(data_dir + "/Y_train.pkl", "rb") as f:
        Y_train = pickle.load(f)
    with open(data_dir + "/X_val.pkl", "rb") as f:
        X_val = pickle.load(f)
    with open(data_dir + "/Y_val.pkl", "rb") as f:
        Y_val = pickle.load(f)

    with open(data_dir + "/max_seq_len.pkl", "rb") as f:
        max_seq_len = pickle.load(f)
    
    ont_read_annotator = ReadAnnotator(max_seq_len, embedding_dim, num_labels, 
                                       num_conv_layers=num_conv_layers, 
                                       num_lstm_layers=num_lstm_layers)
    
    ont_read_annotator.compile_model(learning_rate=0.001, optimizer='adam')
    ont_read_annotator.summary()
    model, history = train_model(X_train, Y_train, X_val, Y_val, ont_read_annotator.model,
                                 epochs=epochs, batch_size=batch_size)

    model.save(model_dir + "model.h5")
    with open(model_dir + '/training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

if __name__ == "__main__":
    app()
