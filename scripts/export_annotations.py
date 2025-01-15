# from .annotate_new_data import annotate_new_data
# from .annotate_new_data import annotate_new_data_parallel
# from .extract_annotated_seqs import extract_annotated_seq_ends
# from .extract_annotated_seqs import extract_annotated_full_length_seqs
# # from .annotate_new_data import distributed_annotate

# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import gc

# # Function to save the current state (bin_name and chunk) for checkpointing
# def save_checkpoint(checkpoint_file, bin_name, chunk):
#     with open(checkpoint_file, "w") as f:
#         f.write(f"{bin_name},{chunk}")

# # Function to load the last checkpoint
# def load_checkpoint(checkpoint_file, start_bin):
#     if os.path.exists(checkpoint_file):
#         with open(checkpoint_file, "r") as f:
#             bin_name, chunk = f.readline().strip().split(",")
#         return bin_name, int(chunk)
#     return start_bin, 1 # If no checkpoint, start from the first chunk

# # Function to process reads in chunks and save results to the same tab-delimited file
# def process_read_ends_in_chunks_and_save(read_ends, original_read_end_names, model, 
#                                          label_binarizer, actual_lengths,
#                                          seq_order,
#                                          end_length, chunk_size,
#                                          output_path="/annotations.tsv"):
#     num_reads = len(read_ends)
#     header_added = False  # Flag to track whether header has been added
#     original_reads_mark = 0

#     for i in range(0, num_reads, chunk_size):
#         chunk_reads = read_ends[i:i+chunk_size]
#         lengths = actual_lengths[original_reads_mark:original_reads_mark + int(chunk_size/2)]
#         chunk_predictions = annotate_new_data(chunk_reads, model)
#         chunk_contiguous_annotated_sequences = extract_annotated_seq_ends(chunk_reads, 
#                                                                           chunk_predictions, 
#                                                                           label_binarizer,
#                                                                           lengths, 
#                                                                           end_length,
#                                                                           seq_order)

#         # Save the results to the same tab-deli mited file after processing each chunk
#         save_annotated_seqs_to_csv(output_path, original_read_end_names[original_reads_mark: original_reads_mark + int(chunk_size/2)],
#                                    chunk_contiguous_annotated_sequences, append, header_added=header_added)

#         original_reads_mark = original_reads_mark + int(chunk_size/2)
        
#         tf.keras.backend.clear_session()
#         gc.collect()

#         # Set the header_added flag to True after the first chunk
#         header_added = True

# # Modified function to process full-length reads in chunks and save
# def process_full_length_reads_in_chunks_and_save(reads, original_read_names, model, 
#                                                  label_binarizer, actual_lengths, 
#                                                  seq_order, chunk_size, bin_name, add_header, 
#                                                  output_path, n_jobs=12, chunk_start=1, 
#                                                  checkpoint_file="checkpoint.txt"):
#     num_reads = len(reads)
#     original_reads_mark = (chunk_start - 1) * chunk_size
#     chunk = chunk_start

#     for i in range(original_reads_mark, num_reads, chunk_size):
#         chunk_reads = reads[i:i+chunk_size]
#         reads_in_chunk = len(chunk_reads)

#         print(f"\nProcessing bin: {bin_name}, chunk: {chunk}")
#         print(f"Processing chunk: number of reads = {reads_in_chunk}\n")

#         output_file = output_path + "/" + str(bin_name) + "_" + str(chunk) + ".tsv"

#         # Update checkpoint
#         save_checkpoint(checkpoint_file, bin_name, chunk)
        
#         chunk = chunk + 1

#         if reads_in_chunk < n_jobs:
#             n_jobs = reads_in_chunk
#         lengths = actual_lengths[i:i + int(chunk_size)]
        
#         if reads_in_chunk < 100:
#             chunk_predictions = annotate_new_data(chunk_reads, model)
#         else:
#             chunk_predictions = annotate_new_data_parallel(chunk_reads, model)
        
#         chunk_contiguous_annotated_sequences = extract_annotated_full_length_seqs(chunk_reads, 
#                                                                                   chunk_predictions, 
#                                                                                   lengths,
#                                                                                   label_binarizer, 
#                                                                                   seq_order,
#                                                                                   n_jobs=n_jobs)

#         # Save the results to the same tab-delimited file after processing each chunk
#         save_annotated_seqs_to_csv(output_file, original_read_names[i: i + int(chunk_size)],
#                                    chunk_contiguous_annotated_sequences, add_header)

#         add_header = False
        
#         # Clean up after each chunk to free memory
#         tf.keras.backend.clear_session()
#         gc.collect()

# # Save Contiguous Annotated Sequences to TSV (Tab-Delimited) file
# def save_annotated_seqs_to_csv(filename, chunk_read_names, 
#                                contiguous_annotated_sequences,
#                                add_header):
#     df_data = []

#     for i, annotated_read in enumerate(contiguous_annotated_sequences):
#         # print(annotated_read)
#         row_data = {'ReadName': chunk_read_names[i]}
#         row_data['read_length'] = annotated_read['read_length']

#         for label, annotations in annotated_read.items():
#             if label != "archtecture" and label != "reason":
#                 if label != 'cDNA':
#                     if 'Starts' in annotations:
#                         row_data[f'{label}_Starts'] = ', '.join(map(str, annotations['Starts']))
#                         row_data[f'{label}_Ends'] = ', '.join(map(str, annotations['Ends']))
#                         row_data[f'{label}_Sequences'] = ', '.join(annotations['Sequences'])

#         if 'cDNA' in annotated_read:
#             row_data['cDNA_Starts'] = ', '.join(map(str, annotated_read['cDNA']['Starts']))
#             row_data['cDNA_Ends'] = ', '.join(map(str, annotated_read['cDNA']['Ends']))
#         else:
#             row_data['cDNA_Starts'] = row_data['cDNA_Ends'] = 'NA'

#         # Add 'architecture' and 'reason' fields
#         row_data['architecture'] = annotated_read['architecture']
#         row_data['reason'] = annotated_read['reason']
#         row_data['orientation'] = annotated_read['orientation']

#         df_data.append(row_data)

#     df = pd.DataFrame(df_data)
    
#     df.to_csv(filename, mode='w', sep='\t', header=add_header, index=False, na_rep='NA')


from .annotate_new_data import annotate_new_data, annotate_new_data_parallel
from .extract_annotated_seqs import extract_annotated_seq_ends, extract_annotated_full_length_seqs
import os
import numpy as np
import polars as pl
import tensorflow as tf
import gc
import logging
import pandas as pd 
from scripts.correct_barcodes import bc_n_demultiplex
from scripts.demultiplex import parallelize_assign_cell_id

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

# Function to process reads in chunks and save results to the same tab-delimited file
def process_read_ends_in_chunks_and_save(read_ends, original_read_end_names, model, 
                                         label_binarizer, actual_lengths,
                                         seq_order, end_length, chunk_size,
                                         output_path="/annotations.tsv"):
    num_reads = len(read_ends)
    header_added = False  # Flag to track whether header has been added
    original_reads_mark = 0

    for i in range(0, num_reads, chunk_size):
        chunk_reads = read_ends[i:i + chunk_size]
        lengths = actual_lengths[original_reads_mark:original_reads_mark + int(chunk_size / 2)]
        chunk_predictions = annotate_new_data(chunk_reads, model)
        chunk_contiguous_annotated_sequences = extract_annotated_seq_ends(chunk_reads, chunk_predictions, label_binarizer, lengths, end_length, seq_order)

        # Save the results to the same tab-delimited file after processing each chunk
        save_annotated_seqs_to_csv(output_path, original_read_end_names[original_reads_mark: original_reads_mark + int(chunk_size / 2)],
                                   chunk_contiguous_annotated_sequences, append=False, header_added=header_added)

        original_reads_mark = original_reads_mark + int(chunk_size / 2)
        
        tf.keras.backend.clear_session()
        gc.collect()

        # Set the header_added flag to True after the first chunk
        header_added = True

# # Modified function to process full-length reads in chunks and save
# def process_full_length_reads_in_chunks_and_save(reads, original_read_names, model, 
#                                                  label_binarizer, actual_lengths, 
#                                                  seq_order, add_header, append,
#                                                  output_file, n_jobs=12):

#     chunk_reads = reads
#     reads_in_chunk = len(chunk_reads)
    
#     print(f"Processing chunk: number of reads = {reads_in_chunk}\n")

#     if reads_in_chunk < n_jobs:
#         n_jobs = reads_in_chunk
#     lengths = actual_lengths
        
#     if reads_in_chunk < 100:
#         chunk_predictions = annotate_new_data(chunk_reads, model)
#     else:
#         chunk_predictions = annotate_new_data_parallel(chunk_reads, model)

#     chunk_contiguous_annotated_sequences = extract_annotated_full_length_seqs(chunk_reads, chunk_predictions, lengths,
#                                                                               label_binarizer, seq_order, n_jobs=n_jobs)

#     # Save the results to the same tab-delimited file after processing each chunk
#     save_annotated_seqs_to_csv(output_file, original_read_names,
#                                chunk_contiguous_annotated_sequences, 
#                                append, add_header)

#     # Clean up after each chunk to free memory
#     tf.keras.backend.clear_session()
#     gc.collect()

# Modified function to process full-length reads in chunks and save
import os

def process_full_length_reads_in_chunks_and_save(reads, original_read_names, model, 
                                                 label_binarizer, cumulative_barcodes_stats,
                                                 actual_lengths, seq_order, add_header, append,
                                                 bin_name, output_dir, barcodes, whitelist_df, 
                                                 threshold, n_jobs=12):
    
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
        
        # for barcode in barcodes:
        #     row_data[f'{barcode}_Sequences'] = ', '.join(map(str, annotated_read[barcode]['Sequences']))

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

    # Process valid reads for barcodes
    column_mapping = {}

    for barcode in barcodes:
        column_mapping[barcode] = barcode

    whitelist_dict = {input_column: whitelist_df[whitelist_column].dropna().unique().tolist() 
                      for input_column, whitelist_column in column_mapping.items()}

    # Process barcodes in parallel
    if not valid_reads_df.empty:
        # corrected_df, cDNA_lengths = process_barcodes(valid_reads_df, list(column_mapping.keys()), 
        #                                               whitelist_dict, threshold, n_jobs)

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

        # corrected_demultplexed_df, match_type_counts, cell_id_counts = parallelize_assign_cell_id(corrected_df, whitelist_df, n_jobs)

        # Save the corrected data to the output file
        # corrected_df.to_csv(valid_output_file, sep='\t', index=False, mode=append, header=add_header)
        # corrected_demultplexed_df.to_csv(valid_output_file, sep='\t', index=False, mode=append, header=add_header)
        corrected_df.to_csv(valid_output_file, sep='\t', index=False, mode=append, header=add_header)
        
        print(f"Processed and saved {len(corrected_df)} valid reads to {valid_output_file}")

        return match_type_counts, cell_id_counts, cDNA_lengths, cumulative_barcodes_stats

    # Clean up after each chunk to free memory
    tf.keras.backend.clear_session()
    gc.collect()

# # Save Contiguous Annotated Sequences to TSV (Tab-Delimited) file
# def save_annotated_seqs_to_csv(filename, chunk_read_names, contiguous_annotated_sequences, append, add_header):
#     df_data = []

#     for i, annotated_read in enumerate(contiguous_annotated_sequences):
#         row_data = {'ReadName': chunk_read_names[i]}
#         row_data['read_length'] = annotated_read['read_length']

#         for label, annotations in annotated_read.items():
#             if label != "architecture" and label != "reason":
#                 if label != 'cDNA':
#                     if 'Starts' in annotations:
#                         row_data[f'{label}_Starts'] = ', '.join(map(str, annotations['Starts']))
#                         row_data[f'{label}_Ends'] = ', '.join(map(str, annotations['Ends']))

#         # Add 'architecture' and 'reason' fields
#         row_data['architecture'] = annotated_read['architecture']
#         row_data['reason'] = annotated_read['reason']
#         row_data['orientation'] = annotated_read['orientation']

#         df_data.append(row_data)

#     df = pl.DataFrame(df_data)

#     with open(filename, mode=append) as f:
#         df.write_csv(f, separator='\t', include_header=add_header)