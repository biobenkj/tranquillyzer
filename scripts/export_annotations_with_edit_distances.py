import os
import re
import gc
import csv
import logging
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import tensorflow as tf
from filelock import FileLock
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein

from matplotlib.backends.backend_pdf import PdfPages
from scripts.correct_barcodes import bc_n_demultiplex
from scripts.extract_annotated_seqs import extract_annotated_full_length_seqs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Edit Distance Helper Functions
# TODO: reorganize into an Annotations class for better portability
# This exists in parallel with the export_annotations.py for testing
# It will be integrated into the existing export_annotations.py once
# it's confirmed to work as intended.

def reverse_complement(seq):
    """
    Reverse complement a DNA sequence.
    Re-defined here, but also exists in simulate_training_data.py
    and also in correct_barcodes.py using a numba approach that is
    jit compiled (faster)
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join([complement.get(base, 'N') for base in seq[::-1]])


def load_whitelist_sequences(whitelist_path):
    """
    Load sequences from whitelist file (tab-separated ID\tSequence).
    These are typically the CBC\ti5\ti7 whitelist/samplesheet
    """
    sequences = []
    if os.path.exists(whitelist_path):
        with open(whitelist_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    sequences.append(parts[1])  # Get sequence column
    return sequences


def calculate_min_edit_distance(detected_seq, reference_seqs, check_revcomp=True):
    """
    Calculate minimum edit distance between detected sequence and reference list.

    Args:
        detected_seq: Detected sequence from read
        reference_seqs: List of reference sequences or single reference sequence
        check_revcomp: If True, also check reverse complement

    Returns:
        Tuple of (min_distance, match_orientation)
          - min_distance: Minimum Levenshtein distance found
          - match_orientation: 'fwd', 'rev', or None
    """
    if not detected_seq or not reference_seqs:
        return (None, None)

    # Handle single reference sequence
    if isinstance(reference_seqs, str):
        reference_seqs = [reference_seqs]

    min_dist = float('inf')
    best_orientation = None

    # Check forward orientation
    for ref_seq in reference_seqs:
        dist = Levenshtein.distance(detected_seq, ref_seq)
        if dist < min_dist:
            min_dist = dist
            best_orientation = 'fwd'

    # Check reverse complement if requested
    if check_revcomp:
        detected_rc = reverse_complement(detected_seq)
        for ref_seq in reference_seqs:
            dist = Levenshtein.distance(detected_rc, ref_seq)
            if dist < min_dist:
                min_dist = dist
                best_orientation = 'rev'

    return (min_dist if min_dist != float('inf') else None, best_orientation)


def load_file_whitelists(whitelist_base_dir=None, cbc_file=None, i5_file=None, i7_file=None):
    """
    Load whitelist sequences from separate files as fallback when not in sample sheet.

    Priority:
    1. If whitelist_base_dir provided, load from standard filenames (cbc.txt, udi_i5.txt, udi_i7.txt)
    2. If individual files provided (cbc_file, i5_file, i7_file), load from those paths

    Args:
        whitelist_base_dir: Base directory containing cbc.txt, udi_i5.txt, udi_i7.txt
        cbc_file: Direct path to CBC whitelist file (fallback)
        i5_file: Direct path to i5 whitelist file (fallback)
        i7_file: Direct path to i7 whitelist file (fallback)

    Returns:
        Dict mapping segment names ('CBC', 'i5', 'i7') to lists of sequences
    """
    whitelists = {}

    # Define file mappings
    if whitelist_base_dir:
        whitelist_files = {
            'CBC': os.path.join(whitelist_base_dir, 'cbc.txt'),
            'i7': os.path.join(whitelist_base_dir, 'udi_i7.txt'),
            'i5': os.path.join(whitelist_base_dir, 'udi_i5.txt'),
        }
    else:
        # Fallback to individual files
        whitelist_files = {
            'CBC': cbc_file,
            'i7': i7_file,
            'i5': i5_file,
        }

    # Load each whitelist
    for segment, filepath in whitelist_files.items():
        if filepath and os.path.exists(filepath):
            whitelists[segment] = load_whitelist_sequences(filepath)
            if whitelists[segment]:
                logger.info(f"Loaded {len(whitelists[segment])} sequences for {segment} from {filepath}")
            else:
                logger.warning(f"No sequences loaded for {segment} from {filepath}")
        else:
            whitelists[segment] = []
            if filepath:
                logger.warning(f"Whitelist file not found for {segment}: {filepath}")

    return whitelists


def parse_seq_orders_for_known_sequences(seq_orders_path, model_name):
    """
    Parse seq_orders.tsv to get known sequences for a specific model.

    Returns:
        Dict mapping segment labels to their known sequences
    """
    known_seqs = {}

    if not os.path.exists(seq_orders_path):
        logger.warning(f"seq_orders file not found: {seq_orders_path}")
        return known_seqs

    with open(seq_orders_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3 and parts[0] == model_name:
                # Parse segment order
                seg_order = parts[1].strip('"').split(',')
                # Parse sequences
                seqs = parts[2].strip('"').split(',')

                for label, seq in zip(seg_order, seqs):
                    # Skip variable length segments (patterns with N)
                    # But include them for reference
                    if seq not in ['NN', 'A', 'T']:
                        known_seqs[label] = seq
                        # Log fixed sequences only (no N's)
                        if 'N' not in seq:
                            logger.info(f"Loaded known sequence for {label}: {seq[:20]}{'...' if len(seq) > 20 else ''}")

                break

    return known_seqs


# Cache for loaded whitelists and sequences
_CACHED_WHITELISTS = None
_CACHED_KNOWN_SEQUENCES = None


def get_or_load_whitelists_and_sequences(seq_orders_path=None, model_name=None, whitelist_base_dir=None):
    """
    Get or load whitelists and known sequences (cached).

    Args:
        seq_orders_path: Path to seq_orders.tsv file
        model_name: Model name to look up in seq_orders.tsv
        whitelist_base_dir: Base directory for whitelist files (optional)

    Returns:
        Tuple of (whitelists_dict, known_sequences_dict)
    """
    global _CACHED_WHITELISTS, _CACHED_KNOWN_SEQUENCES

    # Load whitelists if not cached
    if _CACHED_WHITELISTS is None:
        _CACHED_WHITELISTS = load_file_whitelists(whitelist_base_dir=whitelist_base_dir)

    # Load known sequences if not cached and parameters provided
    if _CACHED_KNOWN_SEQUENCES is None and seq_orders_path and model_name:
        _CACHED_KNOWN_SEQUENCES = parse_seq_orders_for_known_sequences(seq_orders_path, model_name)

    return _CACHED_WHITELISTS, _CACHED_KNOWN_SEQUENCES or {}

# Checkpointing Functions

def save_checkpoint(checkpoint_file, bin_name, chunk):
    with open(checkpoint_file, "w") as f:
        f.write(f"{bin_name},{chunk}")


def load_checkpoint(checkpoint_file, start_bin):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            bin_name, chunk = f.readline().strip().split(",")
        return bin_name, int(chunk)
    return start_bin, 1



# Main Processing Function with Edit Distances

def process_full_length_reads_in_chunks_and_save(reads, original_read_names,
                                                 strand, output_fmt,
                                                 base_qualities,
                                                 model_type, pass_num,
                                                 model_path_w_CRF,
                                                 predictions, bin_name, chunk_idx,
                                                 label_binarizer,
                                                 cumulative_barcodes_stats,
                                                 actual_lengths, seq_order,
                                                 add_header, output_dir,
                                                 invalid_output_file,
                                                 invalid_file_lock,
                                                 valid_output_file,
                                                 valid_file_lock, barcodes,
                                                 whitelist_df, whitelist_dict,
                                                 demuxed_fasta,
                                                 demuxed_fasta_lock,
                                                 ambiguous_fasta,
                                                 ambiguous_fasta_lock,
                                                 threshold, n_jobs,
                                                 seq_orders_path=None,
                                                 model_name=None,
                                                 whitelist_base_dir=None):

    reads_in_chunk = len(reads)
    logging.info(f"Post-processing {bin_name} chunk - {chunk_idx}: number of reads = {reads_in_chunk}")

    n_jobs_extract = min(16, reads_in_chunk)
    chunk_contiguous_annotated_sequences = extract_annotated_full_length_seqs(
        reads, predictions, model_path_w_CRF,
        actual_lengths, label_binarizer, seq_order,
        barcodes, n_jobs_extract
    )

    # Load whitelists and known sequences for edit distance calculation
    segment_whitelists, known_sequences = get_or_load_whitelists_and_sequences(
        seq_orders_path, model_name, whitelist_base_dir
    )

    # Build edit distance columns for barcode segments (with whitelist matching)
    def get_barcode_edit_dist_cols(annotated_read):
        """Get edit distance columns for barcode segments."""
        cols = {}
        for label in ['CBC', 'i7', 'i5']:
            if label in barcodes and label in annotated_read:
                detected_seq = (annotated_read[label]['Sequences'][0]
                               if annotated_read[label].get('Sequences') else "")
                min_dist, orientation = calculate_min_edit_distance(
                    detected_seq,
                    segment_whitelists.get(label, []),
                    check_revcomp=True
                )
                cols[f'{label}_edit_distance'] = min_dist
                cols[f'{label}_match_orientation'] = orientation
        return cols

    # Build edit distance columns for fixed sequences
    def get_fixed_seq_edit_dist_cols(annotated_read):
        """Get edit distance columns for fixed reference sequences."""
        cols = {}
        for label, ref_seq in known_sequences.items():
            # Skip variable-length segments (those with N patterns)
            if 'N' in ref_seq or label in barcodes:
                continue

            if label in annotated_read:
                detected_seq = (annotated_read[label]['Sequences'][0]
                               if annotated_read[label].get('Sequences') else "")
                min_dist, orientation = calculate_min_edit_distance(
                    detected_seq,
                    ref_seq,
                    check_revcomp=True
                )
                cols[f'{label}_edit_distance'] = min_dist
                cols[f'{label}_match_orientation'] = orientation
        return cols

    # DataFrame construction with edit distances
    chunk_df = pd.DataFrame.from_records(
        (
            {
                'ReadName': original_read_names[i],
                'read_length': annotated_read['read_length'],
                'read': annotated_read['read'],

                # Existing columns: Start/End positions
                **{
                    f'{label}_Starts': ', '.join(map(str, annotations['Starts']))
                    for label, annotations in annotated_read.items()
                    if label not in {"architecture", "reason"} and 'Starts' in annotations
                },
                **{
                    f'{label}_Ends': ', '.join(map(str, annotations['Ends']))
                    for label, annotations in annotated_read.items()
                    if label not in {"architecture", "reason"} and 'Ends' in annotations
                },

                # Existing barcode sequences
                **{
                    f'{label}_Sequences': ', '.join(map(str, annotated_read[label]['Sequences']))
                    for label in barcodes
                    if label in annotated_read and 'Sequences' in annotated_read[label]
                },

                # Edit distances for barcode segments (against whitelists)
                **get_barcode_edit_dist_cols(annotated_read),

                # Edit distances for fixed sequences (p7, RP1, RP2, p5, etc.)
                **get_fixed_seq_edit_dist_cols(annotated_read),

                'base_qualities': base_qualities[i] if output_fmt == "fastq" else None,
                'architecture': annotated_read['architecture'],
                'reason': annotated_read['reason'],
                'orientation': annotated_read['orientation']
            }
            for i, annotated_read in enumerate(chunk_contiguous_annotated_sequences)
        )
    )

    # Filter out invalid reads
    invalid_reads_df = chunk_df[chunk_df['architecture'] == 'invalid']
    valid_reads_df = chunk_df[chunk_df['architecture'] != 'invalid']

    if model_type == "HYB" and pass_num == 1:
        tmp_invalid_dir = os.path.join(output_dir, "tmp_invalid_reads")
        os.makedirs(tmp_invalid_dir, exist_ok=True)

        tmp_invalid_df = pl.DataFrame({
            'ReadName': invalid_reads_df['ReadName'],
            'read': invalid_reads_df['read'],
            'read_length': invalid_reads_df['read_length']
        })

        tmp_path = f'{tmp_invalid_dir}/{bin_name}.tsv'
        lock_path = f"{tmp_path}.lock"

        if not os.path.exists(lock_path):
            with open(lock_path, 'w') as lock_file:
                lock_file.write('')

        with FileLock(lock_path):
            write_header = not os.path.exists(tmp_path)
            with open(tmp_path, 'a', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                if write_header:
                    writer.writerow(tmp_invalid_df.columns)
                writer.writerows(tmp_invalid_df.rows())

    else:
        if not invalid_reads_df.empty:
            with invalid_file_lock:
                add_header = not os.path.exists(invalid_output_file) or os.path.getsize(invalid_output_file) == 0
                invalid_reads_df.to_csv(invalid_output_file, sep='\t', index=False, mode='a', header=add_header)

    # Process valid reads for barcodes
    column_mapping = {}
    for barcode in barcodes:
        column_mapping[barcode] = barcode

    # Process barcodes in parallel
    if not valid_reads_df.empty:
        corrected_df, match_type_counts, cell_id_counts = bc_n_demultiplex(
            valid_reads_df, strand,
            list(column_mapping.keys()),
            whitelist_dict, whitelist_df, threshold,
            output_dir, output_fmt, demuxed_fasta,
            demuxed_fasta_lock, ambiguous_fasta,
            ambiguous_fasta_lock, n_jobs
        )

        # Compute barcode stats
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

        # Save valid reads with all edit distance columns
        with valid_file_lock:
            add_header = not os.path.exists(valid_output_file) or os.path.getsize(valid_output_file) == 0
            corrected_df.to_csv(valid_output_file, sep='\t', index=False, mode='a', header=add_header)

        logging.info(f"Post-processed {bin_name} chunk - {chunk_idx}: number of reads = {reads_in_chunk}")

        return match_type_counts, cell_id_counts, cumulative_barcodes_stats

    for local_df in ["chunk_df", "corrected_df", "invalid_reads_df", "valid_reads_df"]:
        if local_df:
            del local_df

    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()


def post_process_reads(reads, read_names, strand, output_fmt,
                       base_qualities, model_type, pass_num,
                       model_path_w_CRF, predictions, label_binarizer,
                       cumulative_barcodes_stats, read_lengths,
                       seq_order, add_header, bin_name, chunk_idx, output_dir,
                       invalid_output_file, invalid_file_lock,
                       valid_output_file, valid_file_lock, barcodes,
                       whitelist_df, whitelist_dict, threshold,
                       checkpoint_file, chunk_start, match_type_counter,
                       cell_id_counter, demuxed_fasta, demuxed_fasta_lock,
                       ambiguous_fasta, ambiguous_fasta_lock, njobs,
                       seq_orders_path=None, model_name=None, whitelist_base_dir=None):

    results = process_full_length_reads_in_chunks_and_save(
        reads, read_names, strand, output_fmt,
        base_qualities, model_type, pass_num,
        model_path_w_CRF, predictions,
        bin_name, chunk_idx,
        label_binarizer, cumulative_barcodes_stats,
        read_lengths, seq_order,
        add_header, output_dir,
        invalid_output_file, invalid_file_lock,
        valid_output_file, valid_file_lock,
        barcodes, whitelist_df, whitelist_dict,
        demuxed_fasta, demuxed_fasta_lock,
        ambiguous_fasta, ambiguous_fasta_lock,
        threshold, njobs,
        seq_orders_path, model_name, whitelist_base_dir
    )

    if results is not None:
        match_type_counts, cell_id_counts, cumulative_barcodes_stats = results

        for key, value in match_type_counts.items():
            match_type_counter[key] += value
        for key, value in cell_id_counts.items():
            cell_id_counter[key] += value

    save_checkpoint(checkpoint_file, bin_name, chunk_start)

    gc.collect()

    return cumulative_barcodes_stats, match_type_counter, cell_id_counter


def filtering_reason_stats(reason_counter_by_bin, output_dir):

    raw_counts_df = pd.DataFrame.from_dict(reason_counter_by_bin, orient='index').fillna(0).T
    total_reads = raw_counts_df.sum(axis=0)
    normalized_data = raw_counts_df.div(total_reads, axis=1)

    raw_counts_df.to_csv(f"{output_dir}/filtered_raw_counts_by_bins.tsv", sep='\t')
    normalized_data.to_csv(f"{output_dir}/filtered_normalized_fractions_by_bins.tsv", sep='\t')

    print(f"Saved raw counts to {output_dir}/filtered_raw_counts_by_bins.tsv")
    print(f"Saved normalized fractions to {output_dir}/filtered_normalized_fractions_by_bins.tsv")


def plot_read_n_cDNA_lengths(output_dir):
    df = pl.read_parquet(f"{output_dir}/annotations_valid.parquet",
                         columns=["read_length", "cDNA_length"])
    read_lengths = []
    cDNA_lengths = []

    read_lengths.extend(df["read_length"].to_list())
    read_lengths = np.array(read_lengths, dtype=int)

    cDNA_lengths.extend(df["cDNA_length"].to_list())
    cDNA_lengths = np.array(cDNA_lengths, dtype=int)

    log_read_lengths = np.log10(read_lengths[read_lengths > 0])
    log_cDNA_lengths = np.log10(cDNA_lengths[cDNA_lengths > 0])

    with PdfPages(f"{output_dir}/plots/cDNA_len_distr.pdf") as pdf:
        if len(log_read_lengths[log_read_lengths > 0]):
            plt.figure(figsize=(8, 6))
            plt.hist(log_read_lengths[log_read_lengths > 0],
                     bins=100, color='blue', edgecolor='black')
            plt.title('Read Length Distribution (Log Scale)')
            plt.xlabel('Log10(Read Length)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        if len(log_cDNA_lengths[log_cDNA_lengths > 0]):
            plt.figure(figsize=(8, 6))
            plt.hist(log_cDNA_lengths[log_cDNA_lengths > 0],
                     bins=100, color='blue', edgecolor='black')
            plt.title('cDNA Length Distribution (Log Scale)')
            plt.xlabel('Log10(cDNA Length)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            pdf.savefig()
            plt.close()
