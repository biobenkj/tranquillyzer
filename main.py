import os
import gc
import time
import json
import typer
import queue
import random
import pickle
import logging
import itertools
import warnings
import subprocess
import numpy as np
import polars as pl
import pandas as pd
import multiprocessing as mp
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
from filelock import FileLock
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from multiprocessing import Manager
from collections import defaultdict
from sklearn.preprocessing import LabelBinarizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_addons as tfa

from scripts.deduplicate import deduplication_parallel
from scripts.train_new_model import DynamicPaddingDataGenerator, ont_read_annotator
from scripts.visualize_annot import save_plots_to_pdf
from scripts.simulate_training_data import generate_training_reads
from scripts.demultiplex import generate_demux_stats_pdf
from scripts.plot_read_len_distr import plot_read_len_distr
from scripts.trained_models import trained_models, seq_orders, training_seq_orders
from scripts.correct_barcodes import generate_barcodes_stats_pdf
from scripts.extract_annotated_seqs import extract_annotated_full_length_seqs
from scripts.annotate_new_data import preprocess_sequences, annotate_new_data, model_predictions, estimate_average_read_length_from_bin
from scripts.preprocess_reads import parallel_preprocess_data, find_sequence_files, extract_and_bin_reads, convert_tsv_to_parquet
from scripts.export_annotations import post_process_reads, plot_read_n_cDNA_lengths

app = typer.Typer()

############# available trained models ################

@app.command()
def availableModels():
    trained_models()

############# extract reads, read_names from fasta file ################

@app.command()
def preprocessFasta(fasta_dir: str, output_dir: str,
                    threads: int = typer.Argument(1),
                    batch_size: int = typer.Argument(100000)):
    
    os.system("mkdir -p " + output_dir + "/full_length_pp_fa")
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
    
# ############# plot read length distribution ################

@app.command()
def readlengthDist(output_dir: str):
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
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
              read_names: str = typer.Option(None, help="Comma-separated list of read names to visualize")):
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    models_dir = os.path.join(base_dir, "models")  # go one level up to 'tranquilizer/', then into 'models'
    models_dir = os.path.abspath(models_dir)

    utils_dir = os.path.join(base_dir, "utils")
    utils_dir = os.path.abspath(utils_dir)

    # model = "models/" + model_name + ".h5"
    model = os.path.join(models_dir, model_name + ".h5")
    model = load_model(model)

    # with open("models/" + model_name + "_lbl_bin.pkl", "rb") as f:
    with open(os.path.join(models_dir, model_name + "_lbl_bin.pkl"), "rb") as f:
        label_binarizer = pickle.load(f)

    # seq_order, sequences, barcodes, UMIs = seq_orders("utils/seq_orders.tsv", model_name)

    seq_order, sequences, barcodes, UMIs = seq_orders(os.path.join(utils_dir, "seq_orders.tsv"), model_name)

    palette = ['red', 'blue', 'green', 'purple', 'pink', 'cyan', 'magenta', 'orange', 'brown']
    colors = {'random_s': 'black', 'random_e': 'black', 'cDNA': 'gray', 'polyT': 'orange', 'polyA': 'orange'}

    i = 0
    for element in seq_order:
        if element not in ['random_s', 'random_e', 'cDNA', 'polyT', 'polyA']:
            colors[element] = palette[i % len(palette)]  # Cycle through the palette
            i += 1

    # Path to the read_index.parquet
    index_file_path = os.path.join(output_dir, "full_length_pp_fa/read_index.parquet")

    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    folder_path = os.path.join(output_dir, "full_length_pp_fa")
    pdf_filename = os.path.join(output_dir, "plots/full_read_annots.pdf")

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
    encoded_data = preprocess_sequences(selected_reads)
    predictions = annotate_new_data(encoded_data, model)
    annotated_reads = extract_annotated_full_length_seqs(
            selected_reads, predictions, selected_read_lengths, label_binarizer, seq_order, barcodes, n_jobs=1
        )
    save_plots_to_pdf(selected_reads, annotated_reads, selected_read_names, pdf_filename, colors, chars_per_line=150)
    
############# Annotate all the reads ################

@app.command()
def annotate_reads(
    model_name: str, 
    output_dir: str, 
    whitelist_file: str,
    chunk_size: int = typer.Option(100000, help="Base chunk size, dynamically adjusts based on read length"),
    bc_lv_threshold: int = typer.Option(2, help="lv-distance threshold for barcode correction"),
    njobs: int = typer.Option(12, help="Number of CPU threads for barcode correction and demultiplexing"),
    max_queue_size: int = typer.Option(3, help="Max number of Parquet files to queue for post-processing")
):
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    models_dir = os.path.join(base_dir, "models")  # go one level up to 'tranquilizer/', then into 'models'
    models_dir = os.path.abspath(models_dir)

    utils_dir = os.path.join(base_dir, "utils")
    utils_dir = os.path.abspath(utils_dir)

    model_path = f"{models_dir}/{model_name}.h5"

    with open(f"{models_dir}/{model_name}_lbl_bin.pkl", "rb") as f:
        label_binarizer = pickle.load(f)

    seq_order, sequences, barcodes, UMIs = seq_orders(f"{utils_dir}/seq_orders.tsv", model_name)
    whitelist_df = pd.read_csv(whitelist_file, sep='\t')

    base_folder_path = os.path.join(output_dir, "full_length_pp_fa")
    
    invalid_output_file = os.path.join(output_dir, "annotations_invalid.tsv")
    valid_output_file = os.path.join(output_dir, "annotations_valid.tsv")

    parquet_files = sorted(
        [os.path.join(base_folder_path, f) for f in os.listdir(base_folder_path) 
         if f.endswith('.parquet') and not f.endswith('read_index.parquet')],
        key=lambda f: estimate_average_read_length_from_bin(os.path.basename(f).replace(".parquet", ""))
    )

    column_mapping = {barcode: barcode for barcode in barcodes}
    
    whitelist_dict = {
    "cell_ids": {
        idx + 1: "-".join(map(str, row.dropna().unique()))
        for idx, row in whitelist_df[list(column_mapping.values())].iterrows()
    },
    **{
        input_column: whitelist_df[whitelist_column].dropna().unique().tolist()
        for input_column, whitelist_column in column_mapping.items()
    }
}
    manager = Manager()
    cumulative_barcodes_stats = manager.dict({barcode: {'count_data': manager.dict(), 'min_dist_data': manager.dict()} for barcode in column_mapping.keys()})
    match_type_counter = manager.dict()
    cell_id_counter = manager.dict()

    task_queue = mp.Queue(maxsize=max_queue_size)
    result_queue = mp.Queue()
    count = mp.Value('i', 0)
    header_track = mp.Value('i', 0)

    fasta_dir = os.path.join(output_dir, "demuxed_fasta")
    os.makedirs(fasta_dir, exist_ok=True)

    demuxed_fasta = os.path.join(output_dir, fasta_dir, "demuxed.fasta")
    demuxed_fasta_lock = FileLock(demuxed_fasta + ".lock")

    ambiguous_fasta = os.path.join(output_dir, fasta_dir, "ambiguous.fasta")
    ambiguous_fasta_lock = FileLock(ambiguous_fasta + ".lock")

    invalid_file_lock = FileLock(invalid_output_file + ".lock")
    valid_file_lock = FileLock(valid_output_file + ".lock")

    def post_process_worker(task_queue, count, header_track, result_queue):
        """Worker function for processing reads and returning results."""
        while True:
            try:
                item = task_queue.get(timeout=10)  
                if item is None:
                    break

                parquet_file, chunk_idx, predictions, read_names, reads, read_lengths = item
                bin_name = os.path.basename(parquet_file).replace(".parquet", "")

                append = "w" if chunk_idx == 1 else "a"

                local_cumulative_stats = {barcode: {'count_data': {}, 'min_dist_data': {}} for barcode in column_mapping.keys()}
                local_match_counter, local_cell_counter = defaultdict(int), defaultdict(int)

                checkpoint_file = os.path.join(output_dir, "annotation_checkpoint.txt")

                with header_track.get_lock():
                    add_header = header_track.value == 0

                result = post_process_reads(
                    reads, read_names, predictions, label_binarizer, local_cumulative_stats,
                    read_lengths, seq_order, add_header, bin_name, output_dir, invalid_output_file, invalid_file_lock,
                    valid_output_file, valid_file_lock, barcodes, whitelist_df, 
                    whitelist_dict, bc_lv_threshold, checkpoint_file, 1, local_match_counter, local_cell_counter, 
                    demuxed_fasta, demuxed_fasta_lock,ambiguous_fasta, ambiguous_fasta_lock, njobs
                )

                if result:
                    local_cumulative_stats, local_match_counter, local_cell_counter = result
                    result_queue.put((local_cumulative_stats, local_match_counter, local_cell_counter, bin_name))

                with count.get_lock():
                    count.value += 1

                gc.collect()
            except queue.Empty:
                pass

    num_workers = min(njobs, mp.cpu_count() - 1)
    workers = [mp.Process(target=post_process_worker, args=(task_queue, count, header_track, result_queue)) for _ in range(num_workers)]

    for worker in workers:
        worker.start()

    for parquet_file in parquet_files:
        for item in model_predictions(parquet_file, 1, chunk_size, model_path):
            task_queue.put(item)
            with header_track.get_lock():
                    header_track.value += 1

    for _ in range(num_workers):
        task_queue.put(None)

    while any(worker.is_alive() for worker in workers) or not result_queue.empty():
        try:
            result = result_queue.get(timeout=5)
            if result:
                local_cumulative_stats, local_match_counter, local_cell_counter, bin_name = result
                
                for key, value in local_match_counter.items():
                    match_type_counter[key] = match_type_counter.get(key, 0) + value
                
                for key, value in local_cell_counter.items():
                    cell_id_counter[key] = cell_id_counter.get(key, 0) + value

                for barcode in local_cumulative_stats.keys():
                    for stat in ["count_data", "min_dist_data"]:
                        for key, value in local_cumulative_stats[barcode][stat].items():
                            cumulative_barcodes_stats[barcode][stat][key] = cumulative_barcodes_stats[barcode][stat].get(key, 0) + value

        except queue.Empty:
            pass 

    for worker in workers:
        worker.join()

    cumulative_barcodes_stats = {k: {'count_data': dict(v['count_data']), 'min_dist_data': dict(v['min_dist_data'])} for k, v in cumulative_barcodes_stats.items()}

    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    logger.info("Generating barcode stats plots")
    generate_barcodes_stats_pdf(cumulative_barcodes_stats, list(column_mapping.keys()), 
                                pdf_filename=f"{output_dir}/plots/barcode_plots.pdf")
    logger.info("Generated barcode stats plots")

    logger.info("Generating demux stats plots")
    generate_demux_stats_pdf(f"{output_dir}/plots/demux_plots.pdf", 
                             f"{output_dir}/matchType_readCount.tsv", 
                             f"{output_dir}/cellId_readCount.tsv",
                             match_type_counter, cell_id_counter)
    logger.info("Generated demux stats plots")

    df = pl.scan_csv(f"{output_dir}/annotations_valid.tsv", separator='\t')
    annotations_valid_parquet_file = f"{output_dir}/annotations_valid.parquet"
    
    logger.info(f"Converting annotations_valid.tsv")
    df.sink_parquet(annotations_valid_parquet_file, compression="snappy", row_group_size=chunk_size)
    logger.info(f"Converted annotations_valid.tsv to annotations_valid.parquet")

    os.system(f"rm {output_dir}/annotations_valid.tsv")
    os.system(f"rm {output_dir}/annotations_valid.tsv.lock")

    with open(f"{output_dir}/annotations_invalid.tsv", 'r') as f:
        header = f.readline().strip().split('\t') 

    dtypes = {col: pl.Utf8 for col in header if col != "read_length"}
    dtypes["read_length"] = pl.Int64 

    # df = pl.scan_csv(f"{output_dir}/annotations_invalid.tsv", separator='\t',
    #                  dtypes={"UMI_Starts": pl.Utf8,  "UMI_Ends": pl.Utf8})
    df = pl.scan_csv(f"{output_dir}/annotations_invalid.tsv", separator='\t', dtypes=dtypes)
    annotations_invalid_parquet_file = f"{output_dir}/annotations_invalid.parquet"
    
    logger.info(f"Converting annotations_invalid.tsv")
    df.sink_parquet(annotations_invalid_parquet_file, compression="snappy", row_group_size=chunk_size)
    logger.info(f"Converted annotations_invalid.tsv to annotations_invalid.parquet")

    os.system(f"rm {output_dir}/annotations_invalid.tsv")
    os.system(f"rm {output_dir}/annotations_invalid.tsv.lock")

    logger.info("Generating valid read length and cDNA length distribution plots")
    plot_read_n_cDNA_lengths(output_dir)
    logger.info("Generated valid read length and cDNA length distribution plots")

    os.system(f"rm {output_dir}/demuxed_fasta/demuxed.fasta.lock")
    os.system(f"rm {output_dir}/demuxed_fasta/ambiguous.fasta.lock")

##################### align inserts to the reference genome #####################

@app.command()
def align(
    input_dir: str,
    ref: str,
    output_dir: str, 
    preset: str = typer.Option(None, help="minimap2 preset"),
    filt_mmap: bool = typer.Option(True, help="whether to filter out secondary alignments for the downstream analysis"),
    mapq: int = typer.Option(15, help="minimap mapq for the alignments to be included for the downstream analysis"),
    threads: int = typer.Option(12, help="number of CPU threads"),
    add_minimap_args: str = typer.Option("", help = "additional minimap2 arguments")):

    fasta_file = os.path.join(input_dir, "demuxed_fasta/demuxed.fasta")

    os.makedirs(f'{output_dir}/aligned_files', exist_ok=True)
    output_bam_dir = os.path.join(output_dir, 'aligned_files')

    output_bam = os.path.join(output_bam_dir,"demuxed_aligned.bam")

    filt_sec_al = "-F 0x100" if filt_mmap else ""

    minimap2_cmd = f"minimap2 -t {threads} -ax {preset} {add_minimap_args} {ref} {fasta_file} | samtools view -h {filt_sec_al} -q {mapq} -@ {threads} | samtools sort -@ {threads} -o {output_bam}"

    logger.info(f"Aligning reads to the reference genome")
    subprocess.run(minimap2_cmd, shell=True, check=True)
    logger.info(f"Alignment completed and sorted BAM saved as {output_bam}")

    logger.info(f"Indexing {output_bam}")
    subprocess.run(f"samtools index {output_bam}", shell=True, check=True)
    logger.info(f"Indexing complete")

################### Deduplication using UMI-tools ##################

@app.command()
def dedup(
    input_dir: str,
    lv_threshold: int = typer.Option(3, help="levenshtein distance threshold for umi similarity"),
    stranded: bool = typer.Option(True, help="if directional or non-directional library"),
    per_cell: bool = typer.Option(True, help="whether to correct umi's per cell basis"),
    threads: int = typer.Option(12, help="number of CPU threads")):

    input_bam = os.path.join(input_dir,"aligned_files/demuxed_aligned.bam")
    out_bam = os.path.join(input_dir, "aligned_files/demuxed_aligned_deduped.bam")
    deduplication_parallel(input_bam, out_bam, lv_threshold, per_cell, threads, stranded)

################## Simulate training dataset #####################

@app.command()
def simulate_data(model_name: str,
                  output_dir: str,
                  num_reads: int = typer.Option(50000, help="number of reads to simulate"),
                  mismatch_rate: float = typer.Option(0.05, help="mismatch rate"),
                  insertion_rate: float = typer.Option(0.05, help="insertion rate"), 
                  deletion_rate: float = typer.Option(0.0612981959469103, help="deletion rate"),
                  min_cDNA: int = typer.Option(100, help="minimum cDNA length"),
                  max_cDNA: int = typer.Option(500, help="maximum cDNA length"),
                  polyT_error_rate: float = typer.Option(0.02, help="error rate within polyT or polyA segments"),
                  max_insertions: float = typer.Option(5, help="maximum number of allowed insertions after a base"),
                  threads: int = typer.Option(2, help="number of CPU threads"), 
                  rc: bool = typer.Option(True, help="whether to include reverse complements of the reads in the training data.\nFinal dataset will contain twice the number of user-specified reads")):
    
    reads = []
    labels = []

    length_range = (min_cDNA, max_cDNA)

    seq_order, sequences, barcodes, UMIs, training_segment_orders = training_seq_orders("utils/training_seq_orders.tsv", model_name)
    seq_order_dict = {}

    for i in range(len(seq_order)):
        seq_order_dict[seq_order[i]] = sequences[i]

    if not training_segment_orders == []:
        for i in range(len(training_segment_orders)):
            training_segment_order = training_segment_orders[i]
            training_segment_pattern = []
            for j in range(len(training_segment_order)):
                training_segment_pattern[j] = seq_order_dict[training_segment_order[j]]
            
            logger.info(f"Generating read type {i}")
            local_reads, local_labels = generate_training_reads(num_reads, mismatch_rate, insertion_rate, deletion_rate,
                                                                polyT_error_rate, max_insertions, training_segment_order, 
                                                                training_segment_pattern, length_range, threads, rc)
            reads.extend(local_reads)
            labels.extend(local_labels)
        logger.info("Finished generating all types of reads")
    else:
        training_segment_order = ["cDNA"]
        training_segment_order.extend(seq_order)
        training_segment_order.append("cDNA")

        training_segment_pattern = ["RN"]
        training_segment_pattern.extend(sequences)
        training_segment_pattern.append("RN")

        logger.info(f"Generating reads")
        reads, labels = generate_training_reads(num_reads, mismatch_rate, insertion_rate, deletion_rate, 
                                                polyT_error_rate, max_insertions, training_segment_order, 
                                                training_segment_pattern, length_range, threads, rc)
        logger.info(f"Finished generating reads")

    os.makedirs(f'{output_dir}/simulated_data', exist_ok=True)

    logger.info("Saving the outputs")
    with open(f'{output_dir}/simulated_data/reads.pkl', 'wb') as reads_pkl:
        pickle.dump(reads, reads_pkl)
    with open(f'{output_dir}/simulated_data/labels.pkl', 'wb') as labels_pkl:
        pickle.dump(labels, labels_pkl)
    logger.info("Outputs saved")
    
################## Train model #################

@app.command()
def train_model(model_name: str,
                output_dir: str,
                num_val_reads: int = typer.Option(10, help="number of reads to simulate"),
                mismatch_rate: float = typer.Option(0.05, help="mismatch rate"),
                insertion_rate: float = typer.Option(0.05, help="insertion rate"), 
                deletion_rate: float = typer.Option(0.0612981959469103, help="deletion rate"),
                min_cDNA: int = typer.Option(100, help="minimum cDNA length"),
                max_cDNA: int = typer.Option(500, help="maximum cDNA length"),
                polyT_error_rate: float = typer.Option(0.02, help="error rate within polyT or polyA segments"),
                max_insertions: float = typer.Option(2, help="maximum number of allowed insertions after a base"),
                threads: int = typer.Option(2, help="number of CPU threads"), 
                rc: bool = typer.Option(True, help="whether to include reverse complements of the reads in the training data.\nFinal dataset will contain twice the number of user-specified reads")):
    
    param_file = f'utils/training_params.tsv'

    with open(f"{output_dir}/simulated_data/reads.pkl", "rb") as r:
        reads = pickle.load(r)

    with open(f"{output_dir}/simulated_data/labels.pkl", "rb") as l:
        labels = pickle.load(l)
    
    df = pd.read_csv(param_file, sep="\t")

    # Ensure the model exists in the file
    if model_name not in df.columns:
        logger.info(f"{model_name} not found in the parameter file.")
        return

    logger.info(f"Extracting parameters for {model_name}")

    # Convert the model's parameter column to a dictionary of lists
    param_dict = {df.iloc[i, 0]: df.iloc[i][model_name].split(",") for i in range(len(df))}

    # Generate all possible combinations of parameters for this model
    param_combinations = list(itertools.product(*param_dict.values()))

    length_range = (min_cDNA, max_cDNA)

    seq_order, sequences, barcodes, UMIs, training_segment_orders = training_seq_orders("utils/training_seq_orders.tsv", model_name)

    print(f"seq orders: {seq_order}")

    validation_segment_order = ["cDNA"]
    validation_segment_order.extend(seq_order)
    validation_segment_order.append("cDNA")

    validation_segment_pattern = ["RN"]
    validation_segment_pattern.extend(sequences)
    validation_segment_pattern.append("RN")

    validation_reads, validation_labels = generate_training_reads(num_val_reads, mismatch_rate, insertion_rate, deletion_rate, 
                                                                  polyT_error_rate, max_insertions, validation_segment_order, 
                                                                  sequences, length_range, threads, rc)
    
    palette = ['red', 'blue', 'green', 'purple', 'pink', 'cyan', 'magenta', 'orange', 'brown']
    colors = {'random_s': 'black', 'random_e': 'black', 'cDNA': 'gray', 'polyT': 'orange', 'polyA': 'orange'}

    i = 0
    for element in seq_order:
        if element not in ['random_s', 'random_e', 'cDNA', 'polyT', 'polyA']:
            colors[element] = palette[i % len(palette)]  # Cycle through the palette
            i += 1

    
    validation_read_names = range(len(validation_reads))
    validation_read_lengths = []

    for validation_read in validation_reads:
        validation_read_lengths.append(len(validation_read))
    
    for idx, param_set in enumerate(param_combinations):
        model_filename = f"{model_name}_{idx}.h5"
        param_filename = f"{model_name}_{idx}_params.json"
        curve_filename = f"{model_name}_{idx}_train_curve.png"

        os.makedirs(f'{output_dir}/{model_name}_{idx}', exist_ok=True)
        # Convert tuple to dictionary
        params = dict(zip(param_dict.keys(), param_set))

        # Extract model parameters
        batch_size = int(params.get("batch_size", 64))
        train_fraction = float(params.get("train_fraction", 0.80))
        vocab_size = int(params["vocab_size"])
        embedding_dim = int(params["embedding_dim"])
        num_labels = int(len(seq_order))
        print(f'num of labels: {num_labels}')
        # num_labels = int(params["num_labels"])
        conv_layers = int(params["conv_layers"])
        conv_filters = int(params["conv_filters"])
        conv_kernel_size = int(params["conv_kernel_size"])
        lstm_layers = int(params["lstm_layers"])
        lstm_units = int(params["lstm_units"])
        bidirectional = params["bidirectional"].lower() == "true"
        attention_heads = int(params["attention_heads"])
        dropout_rate = float(params["dropout_rate"])
        regularization = float(params["regularization"])
        learning_rate = float(params["learning_rate"])
        epochs = int(params["epochs"])

        logger.info(f"Training {model_filename} with parameters: {params}")

        # Save the parameters used in training
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{model_name}_{idx}/{param_filename}", "w") as param_file:
            json.dump(params, param_file, indent=4)

        # Shuffle data
        reads, labels = shuffle(reads, labels)

        unique_labels = list(set([item for sublist in labels for item in sublist]))
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(unique_labels)

        # Save label binarizer
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{model_name}_{idx}/{model_name}_{idx}_lbl_bin.pkl", "wb") as lb_file:
            pickle.dump(label_binarizer, lb_file)

        # Train-validation split
        split_index = int(len(reads) * train_fraction)
        train_reads = reads[:split_index]
        train_labels = labels[:split_index]
        val_reads = reads[split_index:]
        val_labels = labels[split_index:]

        logger.info(f"Training reads: {len(train_reads)}, Validation reads: {len(val_reads)}")
        logger.info(f"Training Label Distribution: {Counter([label for seq in train_labels for label in seq])}")
        logger.info(f"Validation Label Distribution: {Counter([label for seq in val_labels for label in seq])}")

        # Data generators
        train_gen = DynamicPaddingDataGenerator(train_reads, train_labels, batch_size, label_binarizer)
        val_gen = DynamicPaddingDataGenerator(val_reads, val_labels, batch_size, label_binarizer)

        # Multi-GPU strategy
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Number of devices: {strategy.num_replicas_in_sync}")

        with strategy.scope():
            model = ont_read_annotator(
                vocab_size, embedding_dim, num_labels, 
                conv_layers=conv_layers, conv_filters=conv_filters, conv_kernel_size=conv_kernel_size, 
                lstm_layers=lstm_layers, lstm_units=lstm_units, bidirectional=bidirectional, 
                attention_heads=attention_heads, dropout_rate=dropout_rate, regularization=regularization
            )

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()

    
        # Callbacks
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        logger.info(f"Training {model_name}_{idx} with parameters: {params}")
        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[early_stopping, reduce_lr])
        model.save(f"{output_dir}/{model_name}_{idx}/{model_filename}")
        logger.info(f" Model {model_name}_{idx} saved successfully!\n")

        # Plot training & validation accuracy/loss curves
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
        plt.plot(history.history['loss'], label='Train Loss', linestyle='dashed')
        plt.plot(history.history['val_loss'], label='Val Loss', linestyle='dashed')
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.title(f"Training Curve for {model_filename}")
        plt.grid(True)

        # Save training curve
        plt.savefig(f"{output_dir}/{model_name}_{idx}/{curve_filename}")
        plt.close()
        print(f"Training curve saved as {curve_filename}\n")

        gc.collect()

        # Save training curve
        plt.savefig(f"{output_dir}/{model_name}_{idx}/{curve_filename}")
        plt.close()
        logger.info(f"Training curve saved as {curve_filename}\n")

        encoded_data = preprocess_sequences(validation_reads)
        predictions = annotate_new_data(encoded_data, model)
        annotated_reads = extract_annotated_full_length_seqs(
            validation_reads, predictions, validation_read_lengths, label_binarizer, seq_order, barcodes, n_jobs=1
        )
        save_plots_to_pdf(validation_reads, annotated_reads, validation_read_names, 
                          f'{output_dir}/{model_name}_{idx}/{model_name}_{idx}_val_viz.pdf', 
                          colors, chars_per_line=150)
        gc.collect()

if __name__ == "__main__":
    app()