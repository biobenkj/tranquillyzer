# import re
# import random
# import pickle
# import numpy as np
# from Bio import SeqIO
# from multiprocessing import Pool

# ############## introduce errors ##############

# def introduce_errors_with_labels_context(sequence, label, mismatch_rate, insertion_rate, 
#                                          deletion_rate, polyT_error_rate, max_insertions):
#     """
#     Introduce errors into the sequence based on mismatch, insertion, and deletion rates.
#     Error rate adjustments are restricted to polyT regions.
#     """
#     error_sequence, error_labels = [], []
#     for i, (base, lbl) in enumerate(zip(sequence, label)):
#         insertion_count = 0
#         r = np.random.random()
        
#         # Adjust error rates for polyT regions
#         if (lbl == "polyT") or (lbl == 'polyA'):
#             local_mismatch_rate = polyT_error_rate
#             local_insertion_rate = polyT_error_rate
#             local_deletion_rate = polyT_error_rate
#         elif lbl == 'ACC':
#             local_mismatch_rate = 0
#             local_insertion_rate = 0
#             local_deletion_rate = 0
#         else:
#             local_mismatch_rate = mismatch_rate
#             local_insertion_rate = insertion_rate
#             local_deletion_rate = deletion_rate

#         # Apply mismatch
#         if r < local_mismatch_rate:
#             error_sequence.append(np.random.choice([b for b in "ATCG" if b != base]))
#             error_labels.append(lbl)

#         # Apply insertion
#         elif r < local_mismatch_rate + local_insertion_rate:
#             error_sequence.append(base)
#             error_labels.append(lbl)
#             while insertion_count < max_insertions:
#                 error_sequence.append(np.random.choice(list("ATCG")))
#                 error_labels.append(lbl)
#                 insertion_count += 1
#                 if np.random.random() >= local_insertion_rate:
#                     break

#         # Apply deletion
#         elif r < local_mismatch_rate + local_insertion_rate + local_deletion_rate:
#             continue  # Skip appending the base (deletion)

#         # No error
#         else:
#             error_sequence.append(base)
#             error_labels.append(lbl)

#     return "".join(error_sequence), error_labels

# ############## generate segments as per training requirement ###############

# def generate_segment(segment_type, segment_pattern, length_range, transcriptome_records):
#     """
#     Generate a DNA sequence based on segment type and pattern.
#     """
#     if re.match(r"N\d+", segment_pattern):  # Example: "N16" → 16 random bases
#         length = int(segment_pattern[1:])
#         sequence = "".join(np.random.choice(list("ATCG")) for _ in range(length))
#         label = [segment_type] * length
        
#     elif segment_pattern == "NN" and segment_type == "cDNA" :
#         length = np.random.randint(length_range[0], length_range[1])
#         if transcriptome_records:
#             transcript = random.choice(transcriptome_records)
#             transcript_seq = str(transcript.seq)
#         else:
#             transcript_seq = "".join(np.random.choice(list("ATCG")) for _ in range(length))
#         if len(transcript_seq) > length:
#             fragment = transcript_seq[:length] if random.random() < 0.5 else transcript_seq[-length:]
#         else:
#             fragment = transcript_seq
#         sequence = fragment
#         label = ["cDNA"] * len(sequence)

#     elif segment_pattern == "RN" and segment_type == "cDNA" :
#         length = np.random.randint(0, 80)
#         if transcriptome_records:
#             transcript = random.choice(transcriptome_records)
#             transcript_seq = str(transcript.seq)
#         else:
#             transcript_seq = "".join(np.random.choice(list("ATCG")) for _ in range(length))
#         if len(transcript_seq) > length:
#             fragment = transcript_seq[:length] if random.random() < 0.5 else transcript_seq[-length:]
#         else:
#             fragment = transcript_seq
#         sequence = fragment
#         label = ["cDNA"] * len(sequence)

#     elif segment_pattern in ["A", "T"]:  # PolyA or PolyT, length 0-50
#         length = np.random.randint(0, 50)
#         sequence = segment_pattern * length
#         label = ["polyA"] * length if segment_pattern == "A" else ["polyT"] * length
#     else:  # Fixed adapter sequences
#         sequence = segment_pattern
#         label = [segment_type] * len(sequence)

#     return sequence, label

# ############## simulate reads and labels ###############

# def simulate_dynamic_batch_complete(num_reads, length_range, mismatch_rate, insertion_rate,
#                                     deletion_rate, polyT_error_rate, max_insertions, 
#                                     segments_order, segments_patterns, transcriptome_records):
#     """
#     Generate complete reads dynamically based on segment order and patterns.
#     """
#     reads, labels = [], []

#     for _ in range(num_reads):
#         read_segments, label_segments = [], []
        
#         for seg_type, seg_pattern in zip(segments_order, segments_patterns):
#             segment_seq, segment_label = generate_segment(seg_type, seg_pattern, length_range, transcriptome_records)
#             read_segments.append(segment_seq)
#             label_segments.append(segment_label)

#         # Introduce errors in each segment
#         error_read_segments, error_label_segments = [], []
#         for segment, segment_label in zip(read_segments, label_segments):
#             error_segment, error_labels = introduce_errors_with_labels_context(
#                 segment, segment_label, mismatch_rate, insertion_rate, deletion_rate, polyT_error_rate, max_insertions
#             )
#             error_read_segments.append(error_segment)
#             error_label_segments.extend(error_labels)

#         # Combine into single read and label
#         reads.append("".join(error_read_segments))
#         labels.append(error_label_segments)

#     return reads, labels

# def reverse_complement(sequence):
#     """Generate the reverse complement of a DNA sequence."""
#     complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
#     return "".join(complement[base] for base in reversed(sequence))

# def reverse_labels(labels):
#     """Reverse labels to align with the reverse complement of the sequence."""
#     return labels[::-1]

# def simulate_dynamic_batch_complete_wrapper(args):
#     """Wrapper for simulate_dynamic_batch_complete to accept a tuple of arguments."""
#     return simulate_dynamic_batch_complete(*args)

# ################### simulate reads and labels in parallel ################

# # Main balanced dataset generation function
# def generate_training_reads(num_reads, mismatch_rate, insertion_rate, deletion_rate, 
#                             polyT_error_rate, max_insertions, segments_order, 
#                             segments_patterns, length_range, num_processes, rc, transcriptome_records=None):
#     """
#     Generate a balanced dataset by combining first-half, second-half, and complete reads
#     with adjusted cDNA length ranges to balance label representation.
#     """

#     num_complete = num_reads

#     args_complete = (num_complete, length_range, mismatch_rate, insertion_rate, deletion_rate, 
#                      polyT_error_rate, max_insertions, segments_order, segments_patterns, transcriptome_records)

#     # Parallelized read generation
#     with Pool(processes=num_processes) as pool:
#         complete_results = pool.map(simulate_dynamic_batch_complete_wrapper, [args_complete])
    
#     # Combine all reads and labels
#     reads, labels = [], []

#     for local_reads, local_labels in complete_results:
#         reads.extend(local_reads)
#         labels.extend(local_labels)

#     if rc:
#         reverse_complement_reads = [reverse_complement(read) for read in reads]
#         reverse_complement_labels = [reverse_labels(label) for label in labels]
#         reads.extend(reverse_complement_reads)
#         labels.extend(reverse_complement_labels)

#     return reads, labels


import re
import random
import numpy as np
from Bio import SeqIO
from multiprocessing import Pool

############## introduce errors ##############

def introduce_errors_with_labels_context(sequence, label, mismatch_rate, insertion_rate, 
                                         deletion_rate, polyT_error_rate, max_insertions):
    error_sequence, error_labels = [], []
    for base, lbl in zip(sequence, label):
        insertion_count = 0
        r = np.random.random()

        if lbl in ("polyT", "polyA"):
            local_mismatch_rate = polyT_error_rate
            local_insertion_rate = polyT_error_rate
            local_deletion_rate = polyT_error_rate
        elif lbl == 'ACC':
            local_mismatch_rate = 0
            local_insertion_rate = 0
            local_deletion_rate = 0
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

def generate_segment(segment_type, segment_pattern, length_range, transcriptome_records):
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
        length = np.random.randint(0, 50)
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
        sequence = segment_pattern
        label = [segment_type] * len(sequence)
    return sequence, label

############## generate valid read ##############

def generate_valid_read(segments_order, segments_patterns, length_range, transcriptome_records):
    read_segments, label_segments = [], []
    for seg_type, seg_pattern in zip(segments_order, segments_patterns):
        s, l = generate_segment(seg_type, seg_pattern, length_range, transcriptome_records)
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

def generate_invalid_read(segments_order, segments_patterns, length_range, transcriptome_records):
    corruption_type = random.choice([
        "concat", "repeat_adapter_5p", "repeat_adapter_3p", "truncated", "junk_insert", "shuffled"
    ])

    if corruption_type == "concat":
        read1, label1 = generate_valid_read(segments_order, segments_patterns, length_range, transcriptome_records)
        read2, label2 = generate_valid_read(segments_order, segments_patterns, length_range, transcriptome_records)

        # Randomly reverse or reverse complement second read
        strand_flip = random.choices(["none", "reverse", "revcomp"], weights=[0.5, 0.25, 0.25])[0]
        if strand_flip == "reverse":
            read2 = read2[::-1]
            label2 = label2[::-1]
        elif strand_flip == "revcomp":
            read2 = reverse_complement(read2)
            label2 = label2[::-1]  # reverse label direction to match RC

        return read1 + read2, label1 + label2

    elif corruption_type == "repeat_adapter_5p":
        adapter_seq, adapter_label = generate_segment(segments_order[1], segments_patterns[1], length_range, transcriptome_records)
        repeated_adapter = adapter_seq * 3
        repeated_labels = [adapter_label[0]] * len(repeated_adapter)
        read, label = generate_valid_read(segments_order, segments_patterns, length_range, transcriptome_records)
        return repeated_adapter + read, repeated_labels + label
    
    elif corruption_type == "repeat_adapter_3p":
        adapter_seq, adapter_label = generate_segment(segments_order[-2], segments_patterns[-2], length_range, transcriptome_records)
        repeated_adapter = adapter_seq * 3
        repeated_labels = [adapter_label[0]] * len(repeated_adapter)
        read, label = generate_valid_read(segments_order, segments_patterns, length_range, transcriptome_records)
        return repeated_adapter + read, repeated_labels + label

#     elif corruption_type == "truncated":
#         new_order = segments_order.copy()
#         new_patterns = segments_patterns.copy()
#         drop_idx = random.choice(range(len(new_order)))
#         del new_order[drop_idx]
#         del new_patterns[drop_idx]
#         return generate_valid_read(new_order, new_patterns, length_range, transcriptome_records)

############## simulate complete batch ##############

def simulate_dynamic_batch_complete(num_reads, length_range, mismatch_rate, insertion_rate,
                                    deletion_rate, polyT_error_rate, max_insertions, 
                                    segments_order, segments_patterns, transcriptome_records,
                                    invalid_fraction=0.3):
    reads, labels = [], []

    for _ in range(num_reads):
        if np.random.rand() < invalid_fraction:
            sequence, label = generate_invalid_read(segments_order, segments_patterns, length_range, transcriptome_records)
        else:
            sequence, label = generate_valid_read(segments_order, segments_patterns, length_range, transcriptome_records)

        sequence, label = introduce_errors_with_labels_context(
            sequence, label, mismatch_rate, insertion_rate, deletion_rate, polyT_error_rate, max_insertions
        )
        reads.append(sequence)
        labels.append(label)

    return reads, labels

############## multiprocessing ##############

def simulate_dynamic_batch_complete_wrapper(args):
    return simulate_dynamic_batch_complete(*args)

############## main generator ##############

def generate_training_reads(num_reads, mismatch_rate, insertion_rate, deletion_rate, 
                            polyT_error_rate, max_insertions, segments_order, 
                            segments_patterns, length_range, num_processes, rc,
                            invalid_fraction=0.3, transcriptome_records=None):

    args_complete = (
        num_reads, length_range, mismatch_rate, insertion_rate, deletion_rate, 
        polyT_error_rate, max_insertions, segments_order, segments_patterns, 
        transcriptome_records, invalid_fraction
    )

    with Pool(processes=num_processes) as pool:
        complete_results = pool.map(simulate_dynamic_batch_complete_wrapper, [args_complete])

    reads, labels = [], []
    for local_reads, local_labels in complete_results:
        reads.extend(local_reads)
        labels.extend(local_labels)

    if rc:
        reverse_complement_reads = [reverse_complement(read) for read in reads]
        reverse_complement_labels = [reverse_labels(label) for label in labels]
        reads.extend(reverse_complement_reads)
        labels.extend(reverse_complement_labels)

    return reads, labels