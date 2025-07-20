import os
import gc
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from rapidfuzz import process
from filelock import FileLock
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import defaultdict
from rapidfuzz.distance import Levenshtein
from matplotlib.backends.backend_pdf import PdfPages

from scripts.demultiplex import assign_cell_id

def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

# Create ASCII lookup table (0-127 only, avoids excess memory usage)
DNA_COMPLEMENT = np.zeros(128, dtype=np.uint8)  # Only map printable ASCII values
DNA_COMPLEMENT[ord('A')] = ord('T')
DNA_COMPLEMENT[ord('T')] = ord('A')
DNA_COMPLEMENT[ord('G')] = ord('C')
DNA_COMPLEMENT[ord('C')] = ord('G')
DNA_COMPLEMENT[ord('N')] = ord('N')  # Keep 'N' unchanged

@njit
def reverse_complement_numba(seq_ascii):
    rev_comp_arr = DNA_COMPLEMENT[seq_ascii[::-1]]  # Reverse & complement
    return rev_comp_arr

def reverse_complement(seq):
    seq_ascii = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)  # Convert string to ASCII uint8
    rev_comp_ascii = reverse_complement_numba(seq_ascii)  # Call optimized function
    return rev_comp_ascii.tobytes().decode('ascii')  # Convert back to string


def correct_barcode(row, column_name, whitelist, threshold):
    observed_barcode = row[column_name]
    reverse_comp_barcode = reverse_complement(observed_barcode)

    # Get distance scores for observed barcode and reverse complement
    candidates = process.extract(observed_barcode, whitelist, scorer=Levenshtein.distance, limit=5)
    candidates_rev = process.extract(reverse_comp_barcode, whitelist, scorer=Levenshtein.distance, limit=5)

    # Combine results and find minimum distance
    all_matches = candidates + candidates_rev
    min_distance = min(match[1] for match in all_matches)

    # Find all closest barcodes with the same minimum distance
    closest_barcodes = [match[0] for match in all_matches if match[1] == min_distance]

    # Handle threshold & multiple matches
    if min_distance > threshold:
        # if len(closest_barcodes) == 1:
        #     return observed_barcode, closest_barcodes[0], min_distance, 1
        return observed_barcode, "NMF", min_distance, len(closest_barcodes)

    return observed_barcode, ",".join(closest_barcodes), min_distance, len(closest_barcodes)


def write_reads_to_fasta(batch_reads, demuxed_fasta, demuxed_fasta_lock, ambiguous_fasta, ambiguous_fasta_lock):
    for cell_id, reads in batch_reads.items():
        if cell_id == "ambiguous":
            with ambiguous_fasta_lock:
                fasta_file = open(ambiguous_fasta, "a")
        else:
            with demuxed_fasta_lock:
                fasta_file = open(demuxed_fasta, "a")
        
        for header, sequence in reads:
            fasta_file.write(f"{header}\n{sequence}\n")
        
        fasta_file.close()

def process_row(row, barcode_columns, whitelist_dict, whitelist_df, threshold, output_dir):
    result = {
        'ReadName': row['ReadName'],
        'read_length': row['read_length'],
        'cDNA_Starts': row['cDNA_Starts'],
        'cDNA_Ends': row['cDNA_Ends'],
        'cDNA_length': int(row['cDNA_Ends']) - int(row['cDNA_Starts']) + 1,
        'UMI_Starts': row['UMI_Starts'],
        'UMI_Ends': row['UMI_Ends'],
        'random_s_Starts': row['random_s_Starts'],
        'random_s_Ends': row['random_s_Ends'],
        'random_e_Starts': row['random_e_Starts'],
        'random_e_Ends': row['random_e_Ends']
    }

    if 'polyA_Starts' in row and row['polyA_Starts'] != "":
        result['polyA_Starts'] = row['polyA_Starts']
        result['polyA_Ends'] = row['polyA_Ends']
        result['polyA_lengths'] = int(row['polyA_Ends']) - int(row['polyA_Starts']) + 1
    elif 'polyT_Starts' in row and row['polyT_Starts'] != "":
        result['polyA_Starts'] = row['polyT_Starts']
        result['polyA_Ends'] = row['polyT_Ends']
        result['polyA_lengths'] = int(row['polyT_Ends']) - int(row['polyT_Starts']) + 1
    else:
        result['polyA_Starts'] = None
        result['polyA_Ends'] = None
        result['polyA_lengths'] = None

    corrected_barcodes = []
    corrected_barcode_seqs = []

    for barcode_column in barcode_columns:
        whitelist = whitelist_dict[barcode_column]
        corrected_barcode, corrected_seq, min_dist, count = correct_barcode(
            row, barcode_column + "_Sequences", whitelist, threshold
        )
        result[f'corrected_{barcode_column}'] = corrected_seq
        result[f'corrected_{barcode_column}_min_dist'] = min_dist
        result[f'corrected_{barcode_column}_counts_with_min_dist'] = count
        result[f'{barcode_column}_Starts'] = row[f'{barcode_column}_Starts']
        result[f'{barcode_column}_Ends'] = row[f'{barcode_column}_Ends']
        corrected_barcodes.append(f"{barcode_column}:{corrected_seq}")
        corrected_barcode_seqs.append(corrected_seq)

    corrected_barcodes_str = ";".join(corrected_barcodes)
    # corrected_barcode_seqs_str = "-".join(corrected_barcode_seqs)

    result['architecture'] = row['architecture']
    result['reason'] = row['reason']
    result['orientation'] = row['orientation']

    cell_id, local_match_counts, local_cell_counts = assign_cell_id(result, whitelist_df, barcode_columns)
    result['cell_id'] = cell_id

    corrected_barcode_seqs_str = whitelist_dict["cell_ids"][cell_id] if cell_id != "ambiguous" else "ambiguous"

    cDNA_sequence = row['read'][int(row['cDNA_Starts']):int(row['cDNA_Ends']) + 1]
    umi_sequence = row['read'][int(row['UMI_Starts']):int(row['UMI_Ends']) + 1]

    batch_reads = defaultdict(list)
    batch_reads[corrected_barcode_seqs_str].append(
        (f">{row['ReadName']}_{corrected_barcode_seqs_str}_{umi_sequence} cell_id:{cell_id}|Barcodes:{corrected_barcodes_str}|UMI:{umi_sequence}", cDNA_sequence)
    )
    return result, local_match_counts, local_cell_counts, batch_reads

def bc_n_demultiplex(chunk, barcode_columns, whitelist_dict, whitelist_df, threshold, output_dir, 
                     demuxed_fasta, demuxed_fasta_lock, ambiguous_fasta, ambiguous_fasta_lock, num_cores):
    args = [(row, barcode_columns, whitelist_dict, whitelist_df, threshold, output_dir) for _, row in chunk.iterrows()]
    batch_reads = defaultdict(list)
    results = []

    if num_cores > 1:
        with Pool(num_cores) as pool:
            results = list(tqdm(pool.starmap(process_row, args), total=len(chunk), desc="Processing rows"))

    elif num_cores == 1:
         # Loop through each row sequentially instead of using multiprocessing
        for arg in tqdm(args, total=len(chunk), desc="Processing rows (no parallelism)"):
            result = process_row(*arg)
            results.append(result)

    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()

    processed_results = [res[0] for res in results]
    all_match_type_counts = [res[1] for res in results]
    all_cell_id_counts = [res[2] for res in results]

    for res in results:
        for cell_id, reads in res[3].items():
            batch_reads[cell_id].extend(reads)

    write_reads_to_fasta(batch_reads, demuxed_fasta, demuxed_fasta_lock, ambiguous_fasta, ambiguous_fasta_lock)

    match_type_counts = defaultdict(int)
    cell_id_counts = defaultdict(int)

    for match_counts in all_match_type_counts:
        for key, value in match_counts.items():
            match_type_counts[key] += value

    for cell_counts in all_cell_id_counts:
        for key, value in cell_counts.items():
            cell_id_counts[key] += value

    corrected_df = pd.DataFrame(processed_results)

    return corrected_df, match_type_counts, cell_id_counts

# Function to generate side-by-side bar plots for each barcode type and save to a single PDF
def generate_barcodes_stats_pdf(cumulative_barcodes_stats, barcode_columns, pdf_filename="barcode_plots.pdf"):
    
    with PdfPages(pdf_filename) as pdf:
        for barcode_column in barcode_columns:
            count_data = pd.Series(cumulative_barcodes_stats[barcode_column]['count_data']).sort_index()
            min_dist_data = pd.Series(cumulative_barcodes_stats[barcode_column]['min_dist_data']).sort_index()

            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            axs[0].bar(count_data.index, count_data.values, color='skyblue')
            axs[0].set_xlabel(f'Number of Matches')
            axs[0].set_ylabel('Frequency')
            axs[0].set_title(f'{barcode_column} - Number of Matches')

            axs[1].bar(min_dist_data.index, min_dist_data.values, color='lightgreen')
            axs[1].set_xlabel(f'Minimum Distance')
            axs[1].set_ylabel('Frequency')
            axs[1].set_title(f'{barcode_column} - Minimum Distance')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()