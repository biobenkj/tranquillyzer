from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method
from .export_annotations import process_chunk, save_annotated_seqs_to_csv

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import gc

def process_reads_in_chunks_and_save(reads, original_read_names, model, label_binarizer, num_cores, chunk_size=10000, output_path="/annotations.tsv"):
    num_reads = len(reads)

    header_added = False  # Flag to track whether header has been added

    # Set the start method to 'spawn' to work around TensorFlow limitations
    set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Divide the work into chunks and process each chunk in parallel
        futures = []
        for i in range(0, num_reads, chunk_size):
            chunk_reads = reads[i:i+chunk_size]
            future = executor.submit(process_chunk, chunk_reads, model, label_binarizer)
            futures.append((future, original_read_names[i:i+chunk_size]))

        # Save the results to the same tab-delimited file after processing each chunk
        for future, original_read_names_chunk in futures:
            chunk_contiguous_annotated_sequences = future.result()
            save_annotated_seqs_to_csv(output_path, original_read_names_chunk, chunk_contiguous_annotated_sequences, append=True, header_added=header_added)
            header_added = True  # Set the header_added flag to True after the first chunk

# Example usage:
# process_reads_in_chunks_and_save_parallel(your_reads, your_original_read_names, your_model, your_label_binarizer, num_cores=32)
