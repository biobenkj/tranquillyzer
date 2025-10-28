import re
import random
import os
import logging
import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter
from multiprocessing import Pool

############## logging #######################

logger = logging.getLogger(__name__)

############## acc priors ####################

def load_acc_priors(utils_dir, model_name):
    """
    Load ACC sequence priors for a specific model from acc_priors.tsv.

    Args:
        utils_dir: Path to utils directory containing acc_priors.tsv
        model_name: Name of the model to get priors for

    Returns:
        Tuple of (acc_sequences, acc_frequencies) or (None, None) if not found
    """
    priors_file = os.path.join(utils_dir, "acc_priors.tsv")

    if not os.path.exists(priors_file):
        logger.info(f"No acc_priors.tsv found in {utils_dir}")
        return None, None

    try:
        df = pd.read_csv(priors_file, sep='\t', comment='#')

        # Filter for this specific model
        model_df = df[df['model_name'] == model_name].copy()

        if model_df.empty:
            logger.info(f"No ACC priors found for model '{model_name}' in acc_priors.tsv")
            return None, None

        # Check if model is set to skip priors (N/A marker)
        if (model_df['sequence'] == 'N/A').any():
            logger.info(f"Model '{model_name}' configured to skip ACC priors (using random generation)")
            return None, None

        sequences = model_df['sequence'].tolist()
        frequencies = model_df['frequency'].tolist()

        # Validate frequencies
        freq_sum = sum(frequencies)
        if abs(freq_sum - 1.0) > 0.01:
            logger.warning(
                f"ACC frequencies sum to {freq_sum:.4f}, normalizing to 1.0"
            )
            frequencies = [f / freq_sum for f in frequencies]

        # Validate sequences are not empty
        if not sequences or any(not s or pd.isna(s) for s in sequences):
            logger.error("Found empty or NaN ACC sequences in priors file")
            return None, None

        logger.info(f"✅ Loaded {len(sequences)} ACC variants for model '{model_name}'")
        logger.info(f"   Sequences: {sequences[:5]}{'...' if len(sequences) > 5 else ''}")
        logger.info(
            f"   Top frequencies: {[f'{f:.3f}' for f in frequencies[:5]]}"
            f"{'...' if len(frequencies) > 5 else ''}"
        )

        return sequences, frequencies

    except Exception as e:
        logger.error(f"Error loading ACC priors from {priors_file}: {e}")
        return None, None


def validate_acc_against_iupac(acc_sequences, iupac_definition):
    """
    Validate that ACC sequences match the IUPAC definition.

    Args:
        acc_sequences: List of ACC sequences from priors
        iupac_definition: IUPAC pattern (e.g., "ACCSSV")

    Returns:
        Tuple of (valid, invalid) sequence lists
    """
    IUPAC_CODES = {
        'S': ['G', 'C'],
        'V': ['A', 'C', 'G'],
        'W': ['A', 'T'],
        'R': ['A', 'G'],
        'Y': ['C', 'T'],
        'M': ['A', 'C'],
        'K': ['G', 'T'],
        'H': ['A', 'C', 'T'],
        'B': ['C', 'G', 'T'],
        'D': ['A', 'G', 'T'],
        'N': ['A', 'C', 'G', 'T']
    }

    valid = []
    invalid = []

    for seq in acc_sequences:
        if len(seq) != len(iupac_definition):
            invalid.append(seq)
            continue

        is_valid = True
        for i, (seq_base, pattern_base) in enumerate(zip(seq, iupac_definition)):
            if pattern_base in IUPAC_CODES:
                if seq_base not in IUPAC_CODES[pattern_base]:
                    is_valid = False
                    break
            else:
                if seq_base != pattern_base:
                    is_valid = False
                    break

        if is_valid:
            valid.append(seq)
        else:
            invalid.append(seq)

    return valid, invalid


def expand_iupac_sequence(seq):
    """
    Expand a single sequence with IUPAC degenerate codes by random sampling.

    Args:
        seq: Sequence string with IUPAC codes (e.g., "ACCSSV")

    Returns:
        One randomly sampled expansion
    """
    IUPAC_CODES = {
        'S': ['G', 'C'],
        'V': ['A', 'C', 'G'],
        'W': ['A', 'T'],
        'R': ['A', 'G'],
        'Y': ['C', 'T'],
        'M': ['A', 'C'],
        'K': ['G', 'T'],
        'H': ['A', 'C', 'T'],
        'B': ['C', 'G', 'T'],
        'D': ['A', 'G', 'T'],
        'N': ['A', 'C', 'G', 'T']
    }

    result = []
    for base in seq:
        if base in IUPAC_CODES:
            result.append(random.choice(IUPAC_CODES[base]))
        else:
            result.append(base)

    return ''.join(result)


def generate_acc_sequence(acc_element_def, acc_priors=None):
    """
    Generate ACC sequence either from priors or IUPAC expansion.

    Args:
        acc_element_def: The sequence definition from training_seq_orders.tsv
                        (e.g., "ACCSSV" or "ACCNNN")
        acc_priors: Tuple of (sequences, frequencies) or None

    Returns:
        Generated ACC sequence string
    """
    if acc_priors is not None:
        sequences, frequencies = acc_priors
        # Sample from the prior distribution
        return random.choices(sequences, weights=frequencies)[0]
    else:
        # Fall back to IUPAC expansion with uniform distribution
        return expand_iupac_sequence(acc_element_def)


def add_sequencing_errors(sequence, error_rate=0.02, indel_rate=0.005):
    """
    Add realistic sequencing errors to a sequence.

    Args:
        sequence: Input DNA sequence
        error_rate: Substitution error rate per base
        indel_rate: Insertion/deletion rate per base

    Returns:
        Sequence with errors introduced
    """
    if error_rate <= 0 and indel_rate <= 0:
        return sequence

    bases = ['A', 'C', 'G', 'T']
    result = []

    for base in sequence:
        # Deletion
        if indel_rate > 0 and random.random() < indel_rate:
            continue  # Skip this base (deletion)

        # Substitution
        if error_rate > 0 and random.random() < error_rate:
            result.append(random.choice([b for b in bases if b != base]))
        else:
            result.append(base)

        # Insertion
        if indel_rate > 0 and random.random() < indel_rate:
            result.append(random.choice(bases))

    return ''.join(result)



############## introduce errors ##############


def introduce_errors_with_labels_context(sequence, label, mismatch_rate,
                                        insertion_rate, deletion_rate,
                                        polyT_error_rate, max_insertions):

    error_sequence, error_labels = [], []
    for base, lbl in zip(sequence, label):
        insertion_count = 0
        r = np.random.random()

        # Higher error rate for polyT regions
        if lbl == "polyT":
            local_mismatch_rate = polyT_error_rate
            local_insertion_rate = polyT_error_rate
            local_deletion_rate = polyT_error_rate
        else:
            local_mismatch_rate = mismatch_rate
            local_insertion_rate = insertion_rate
            local_deletion_rate = deletion_rate

        if r < local_mismatch_rate:
            error_sequence.append(np.random.choice([b for b in "ATCG" if b != base]))
            error_labels.append(lbl)
        elif r < local_mismatch_rate + local_insertion_rate:
            error_sequence.append(base)
            error_labels.append(lbl)
            while insertion_count < max_insertions:
                error_sequence.append(np.random.choice(list("ATCG")))
                error_labels.append(lbl)
                insertion_count += 1
                if np.random.random() >= local_insertion_rate:
                    break
        elif r < local_mismatch_rate + local_insertion_rate + local_deletion_rate:
            continue
        else:
            error_sequence.append(base)
            error_labels.append(lbl)

    return "".join(error_sequence), error_labels

############## generate segments ##############


def generate_segment(segment_type, segment_pattern,
                    length_range, transcriptome_records,
                    acc_priors=None):
    """
    Generate a segment of a read with optional ACC priors support.
    
    Args:
        segment_type: Type of segment (e.g., 'ACC', 'UMI', 'cDNA')
        segment_pattern: Pattern definition (e.g., 'ACCSSV', 'N8', 'NN')
        length_range: Tuple of (min, max) length for variable segments
        transcriptome_records: List of transcript records for cDNA generation
        acc_priors: Optional tuple of (sequences, frequencies) for ACC element
    
    Returns:
        Tuple of (sequence, label)
    """
    
    # Handle ACC element with priors
    if segment_type == 'ACC':
        if acc_priors is not None:
            sequence = generate_acc_sequence(segment_pattern, acc_priors)
            label = [segment_type] * len(sequence)
            return sequence, label
        else:
            # Fall back to IUPAC expansion
            sequence = expand_iupac_sequence(segment_pattern)
            label = [segment_type] * len(sequence)
            return sequence, label
    
    # Original logic for other segment types
    if re.match(r"N\d+", segment_pattern):
        length = int(segment_pattern[1:])
        sequence = "".join(np.random.choice(list("ATCG")) for _ in range(length))
        label = [segment_type] * length
    elif segment_pattern == "NN" and segment_type == "cDNA":
        length = np.random.randint(length_range[0], length_range[1])
        if transcriptome_records:
            transcript = random.choice(transcriptome_records)
            transcript_seq = str(transcript.seq)
        else:
            transcript_seq = "".join(np.random.choice(list("ATCG")) for _ in range(length))
        fragment = transcript_seq[:length] if len(transcript_seq) > length and random.random() < 0.5 else transcript_seq[-length:]
        sequence = fragment
        label = ["cDNA"] * len(sequence)
    elif segment_pattern == "RN" and segment_type == "cDNA":
        length = np.random.randint(5, 50)
        if transcriptome_records:
            transcript = random.choice(transcriptome_records)
            transcript_seq = str(transcript.seq)
        else:
            transcript_seq = "".join(np.random.choice(list("ATCG")) for _ in range(length))
        fragment = transcript_seq[:length] if len(transcript_seq) > length and random.random() < 0.5 else transcript_seq[-length:]
        sequence = fragment
        label = ["cDNA"] * len(sequence)
    elif segment_pattern in ["A", "T"]:
        length = np.random.randint(0, 50)
        sequence = segment_pattern * length
        label = ["polyA"] * length if segment_pattern == "A" else ["polyT"] * length
    else:
        # Check if pattern contains IUPAC codes
        if any(c in segment_pattern for c in 'SVRYWMKHBDN'):
            sequence = expand_iupac_sequence(segment_pattern)
        else:
            sequence = segment_pattern
        label = [segment_type] * len(sequence)
    
    return sequence, label

############## generate valid read ##############


def generate_valid_read(segments_order, segments_patterns,
                       length_range, transcriptome_records,
                       acc_priors=None):
    """
    Generate a valid read with optional ACC priors support.
    
    Args:
        segments_order: List of segment types in order
        segments_patterns: List of segment patterns
        length_range: Tuple of (min, max) for variable length segments
        transcriptome_records: Transcript records for cDNA
        acc_priors: Optional tuple of (sequences, frequencies) for ACC
    
    Returns:
        Tuple of (sequence, labels)
    """
    read_segments, label_segments = [], []
    for seg_type, seg_pattern in zip(segments_order, segments_patterns):
        s, l = generate_segment(seg_type, seg_pattern,
                               length_range,
                               transcriptome_records,
                               acc_priors)
        read_segments.append(s)
        label_segments.append(l)
    return "".join(read_segments), [lbl for seg_lbls in label_segments for lbl in seg_lbls]

############## reverse complement utilities ##############


def reverse_complement(sequence):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(complement[base] for base in reversed(sequence))


def reverse_labels(labels):
    return labels[::-1]

############## generate invalid read with multiple corruption types ##############


def generate_invalid_read(segments_order, segments_patterns,
                         length_range, transcriptome_records,
                         acc_priors=None):
    """
    Generate an invalid read with corruption, with optional ACC priors support.
    
    Args:
        segments_order: List of segment types in order
        segments_patterns: List of segment patterns
        length_range: Tuple of (min, max) for variable length segments
        transcriptome_records: Transcript records for cDNA
        acc_priors: Optional tuple of (sequences, frequencies) for ACC
    
    Returns:
        Tuple of (sequence, labels)
    """
    # corruption_type = random.choice([
    #     "concat", "repeat_adapter_5p",
    #     "repeat_adapter_3p",
    # ])

    corruption_type = random.choice([
        "concat"
    ])

    if corruption_type == "concat":
        read1, label1 = generate_valid_read(segments_order,
                                           segments_patterns,
                                           length_range,
                                           transcriptome_records,
                                           acc_priors)
        read2, label2 = generate_valid_read(segments_order,
                                           segments_patterns,
                                           length_range,
                                           transcriptome_records,
                                           acc_priors)

        # Randomly reverse or reverse complement second read
        strand_flip = random.choices(["none", "reverse", "revcomp"],
                                    weights=[0.5, 0.25, 0.25])[0]

        if strand_flip == "reverse":
            read2 = read2[::-1]
            label2 = label2[::-1]
        if strand_flip == "revcomp":
            read2 = reverse_complement(read2)
            label2 = label2[::-1]  # reverse label direction to match RC

        return read1 + read2, label1 + label2

    elif corruption_type == "repeat_adapter_5p":
        adapter_seq, adapter_label = generate_segment(segments_order[1],
                                                     segments_patterns[1],
                                                     length_range,
                                                     transcriptome_records,
                                                     acc_priors)
        repeated_adapter = adapter_seq * 3
        repeated_labels = [adapter_label[0]] * len(repeated_adapter)
        read, label = generate_valid_read(segments_order,
                                         segments_patterns,
                                         length_range,
                                         transcriptome_records,
                                         acc_priors)
        return repeated_adapter + read, repeated_labels + label

    elif corruption_type == "repeat_adapter_3p":
        adapter_seq, adapter_label = generate_segment(segments_order[-2],
                                                     segments_patterns[-2],
                                                     length_range,
                                                     transcriptome_records,
                                                     acc_priors)
        repeated_adapter = adapter_seq * 3
        repeated_labels = [adapter_label[0]] * len(repeated_adapter)
        read, label = generate_valid_read(segments_order,
                                         segments_patterns,
                                         length_range,
                                         transcriptome_records,
                                         acc_priors)
        return repeated_adapter + read, repeated_labels + label


############## simulate complete batch ##############


def simulate_dynamic_batch_complete(
    num_reads, length_range,
    mismatch_rate, insertion_rate, deletion_rate,
    polyT_error_rate, max_insertions,
    segments_order, segments_patterns,
    transcriptome_records, invalid_fraction, rc,
    acc_priors=None
):
    """
    Simulate a batch of reads with optional ACC priors support.
    
    Args:
        num_reads: Number of reads to generate
        length_range: Tuple of (min, max) for variable segments
        mismatch_rate: Substitution error rate
        insertion_rate: Insertion error rate
        deletion_rate: Deletion error rate
        polyT_error_rate: Error rate for polyT regions
        max_insertions: Maximum consecutive insertions
        segments_order: List of segment types
        segments_patterns: List of segment patterns
        transcriptome_records: Transcript records for cDNA
        invalid_fraction: Fraction of invalid reads to generate
        rc: Whether to generate reverse complement
        acc_priors: Optional tuple of (sequences, frequencies) for ACC
    
    Returns:
        Tuple of (reads, labels)
    """
    reads, labels = [], []

    for _ in range(num_reads):
        # Generate clean read first
        if np.random.rand() < invalid_fraction:
            sequence, label = generate_invalid_read(
                segments_order, segments_patterns,
                length_range, transcriptome_records,
                acc_priors
            )
        else:
            sequence, label = generate_valid_read(
                segments_order, segments_patterns,
                length_range, transcriptome_records,
                acc_priors
            )

        # Add both orientations BEFORE adding noise
        read_pairs = [(sequence, label)]

        if rc:
            rc_seq = reverse_complement(sequence)
            rc_lbl = reverse_labels(label)
            read_pairs.append((rc_seq, rc_lbl))

        # Introduce noise to each orientation
        for seq, lbl in read_pairs:
            seq_err, lbl_err = introduce_errors_with_labels_context(
                seq, lbl, mismatch_rate, insertion_rate,
                deletion_rate, polyT_error_rate, max_insertions
            )
            reads.append(seq_err)
            labels.append(lbl_err)

    return reads, labels

# ############## multiprocessing ##############


def simulate_dynamic_batch_complete_wrapper(args):
    return simulate_dynamic_batch_complete(*args)


# ############## main generator ##############


def generate_training_reads(
    num_reads, mismatch_rate, insertion_rate, deletion_rate,
    polyT_error_rate, max_insertions,
    segments_order, segments_patterns,
    length_range, num_processes, rc,
    transcriptome_records, invalid_fraction,
    acc_priors=None
):
    """
    Generate training reads with optional ACC priors support.
    
    This is the main entry point that maintains backward compatibility
    with the original tranquillyzer implementation while adding ACC priors.
    
    Args:
        num_reads: Number of reads to generate
        mismatch_rate: Substitution error rate
        insertion_rate: Insertion error rate
        deletion_rate: Deletion error rate
        polyT_error_rate: Error rate for polyT regions
        max_insertions: Maximum consecutive insertions
        segments_order: List of segment types in order
        segments_patterns: List of segment patterns
        length_range: Tuple of (min, max) for variable segments
        num_processes: Number of parallel processes
        rc: Whether to generate reverse complement
        transcriptome_records: Transcript records for cDNA generation
        invalid_fraction: Fraction of invalid reads
        acc_priors: Optional tuple of (sequences, frequencies) for ACC element
    
    Returns:
        Tuple of (reads, labels)
    """
    args_complete = (
        num_reads, length_range, mismatch_rate, insertion_rate, deletion_rate,
        polyT_error_rate, max_insertions,
        segments_order, segments_patterns,
        transcriptome_records, invalid_fraction, rc,
        acc_priors  # Pass ACC priors through
    )

    # Parallel or single-thread execution
    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            complete_results = pool.map(
                simulate_dynamic_batch_complete_wrapper, [args_complete]
            )
            pool.close()
    else:
        complete_results = [simulate_dynamic_batch_complete_wrapper(args_complete)]

    reads, labels = [], []
    for local_reads, local_labels in complete_results:
        reads.extend(local_reads)
        labels.extend(local_labels)

    return reads, labels
