import os
import tensorflow as tf
import gc
import logging
import pandas as pd 
from scripts.correct_barcodes import bc_n_demultiplex
from scripts.annotate_new_data import annotate_new_data, annotate_new_data_parallel
from scripts.extract_annotated_seqs import extract_annotated_full_length_seqs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to save the current state (bin_name and chunk) for checkpointing
def save_checkpoint(checkpoint_file, bin_name, chunk):
    with open(checkpoint_file, "w") as f:
        f.write(f"{bin_name},{chunk}")

# Function to load the last checkpoint
def load_checkpoint(checkpoint_file, start_bin):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            bin_name, chunk = f.readline().strip().split(",")
        return bin_name, int(chunk)
    return start_bin, 1  # If no checkpoint, start from the first chunk

def process_full_length_reads_in_chunks_and_save(reads, original_read_names, model, 
                                                 label_binarizer, cumulative_barcodes_stats,
                                                 reason_counter, actual_lengths, seq_order, 
                                                 add_header, append, bin_name, output_dir, 
                                                 barcodes, whitelist_df, threshold, n_jobs=12):
    
    output_path = os.path.join(output_dir, "tmp")
    output_file = f"{output_path}/{bin_name}.tsv"

    # Remove .tsv from output_file if present
    if output_file.endswith(".tsv"):
        output_file = output_file[:-4]

    # File for invalid reads
    invalid_output_file = f"{output_file}_invalid.tsv"
    valid_output_file = f"{output_file}_valid.tsv"

    chunk_reads = reads
    reads_in_chunk = len(chunk_reads)
    
    print(f"Processing chunk: number of reads = {reads_in_chunk}\n")

    if reads_in_chunk < n_jobs:
        n_jobs = reads_in_chunk
    lengths = actual_lengths
        
    # Annotate reads
    if reads_in_chunk < 100:
        chunk_predictions = annotate_new_data(chunk_reads, model)
    else:
        chunk_predictions = annotate_new_data_parallel(chunk_reads, model)

    chunk_contiguous_annotated_sequences = extract_annotated_full_length_seqs(chunk_reads, chunk_predictions, 
                                                                              lengths, label_binarizer, seq_order, 
                                                                              barcodes, n_jobs=n_jobs)

    # Combine ReadName with the annotated sequences
    chunk_data = []
    for i, annotated_read in enumerate(chunk_contiguous_annotated_sequences):
        row_data = {'ReadName': original_read_names[i]}
        row_data['read_length'] = annotated_read['read_length']
        row_data['read'] = annotated_read['read']

        for label, annotations in annotated_read.items():
            if label != "architecture" and label != "reason":
                if 'Starts' in annotations:
                    row_data[f'{label}_Starts'] = ', '.join(map(str, annotations['Starts']))
                    row_data[f'{label}_Ends'] = ', '.join(map(str, annotations['Ends']))
                if label in barcodes:
                    row_data[f'{label}_Sequences'] = ', '.join(map(str, annotated_read[label]['Sequences']))

        # Add 'architecture' and 'reason' fields
        row_data['architecture'] = annotated_read['architecture']
        row_data['reason'] = annotated_read['reason']
        row_data['orientation'] = annotated_read['orientation']

        chunk_data.append(row_data)

    # Convert chunk_data into a DataFrame for processing
    chunk_df = pd.DataFrame(chunk_data)

    # Filter out invalid reads
    invalid_reads_df = chunk_df[chunk_df['architecture'] == 'invalid']
    valid_reads_df = chunk_df[chunk_df['architecture'] != 'invalid']

    # Save invalid reads to a separate file
    if not invalid_reads_df.empty:
        invalid_reads_df.to_csv(invalid_output_file, sep='\t', index=False, mode='a', header=add_header)
        print(f"Saved {len(invalid_reads_df)} invalid reads to {invalid_output_file}")

        for reason in invalid_reads_df["reason"]:
            if reason not in reason_counter:
                reason_counter[reason] += 1
            else:
                reason_counter[reason] += 1

    # Process valid reads for barcodes
    column_mapping = {}

    for barcode in barcodes:
        column_mapping[barcode] = barcode

    whitelist_dict = {input_column: whitelist_df[whitelist_column].dropna().unique().tolist() 
                      for input_column, whitelist_column in column_mapping.items()}

    # Process barcodes in parallel
    if not valid_reads_df.empty:

        corrected_df, cDNA_lengths, match_type_counts, cell_id_counts = bc_n_demultiplex(valid_reads_df, list(column_mapping.keys()), 
                                                                                         whitelist_dict, whitelist_df, threshold, 
                                                                                         output_dir, n_jobs)

        for barcode in list(column_mapping.keys()):
            count_column = f'corrected_{barcode}_counts_with_min_dist'
            min_dist_column = f'corrected_{barcode}_min_dist'

            # Update count stats
            chunk_count_data = corrected_df[count_column].value_counts()
            for key, value in chunk_count_data.items():
                cumulative_barcodes_stats[barcode]['count_data'][key] = (
                    cumulative_barcodes_stats[barcode]['count_data'].get(key, 0) + value
                )

            # Update min distance stats
            chunk_min_dist_data = corrected_df[min_dist_column].value_counts()
            for key, value in chunk_min_dist_data.items():
                cumulative_barcodes_stats[barcode]['min_dist_data'][key] = (
                    cumulative_barcodes_stats[barcode]['min_dist_data'].get(key, 0) + value
                )

        corrected_df.to_csv(valid_output_file, sep='\t', index=False, mode=append, header=add_header)
        
        print(f"Processed and saved {len(corrected_df)} valid reads to {valid_output_file}")

        return match_type_counts, cell_id_counts, cDNA_lengths, cumulative_barcodes_stats, reason_counter

    # Clean up after each chunk to free memory
    tf.keras.backend.clear_session()
    gc.collect()

