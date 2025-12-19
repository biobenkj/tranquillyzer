import re
import random
import numpy as np
from Bio import SeqIO
from multiprocessing import Pool

############## introduce errors ##############


def introduce_errors_with_labels_context(sequence, label, mismatch_rate,
                                         insertion_rate, deletion_rate,
                                         polyT_error_rate, max_insertions):

    error_sequence, error_labels = [], []
    for base, lbl in zip(sequence, label):
        insertion_count = 0
        r = np.random.random()

        if lbl in ("polyT", "polyA"):
            local_mismatch_rate = polyT_error_rate
            local_insertion_rate = polyT_error_rate
            local_deletion_rate = polyT_error_rate
        elif lbl == 'ACC' or lbl == 'cDNA':
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


def generate_segment(segment_type, segment_pattern,
                     length_range, transcriptome_records):

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


def generate_valid_read(segments_order, segments_patterns,
                        length_range, transcriptome_records):
    read_segments, label_segments = [], []
    for seg_type, seg_pattern in zip(segments_order, segments_patterns):
        s, l = generate_segment(seg_type, seg_pattern,
                                length_range,
                                transcriptome_records)
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


# def generate_invalid_read(segments_order, segments_patterns,
#                           length_range, transcriptome_records):
#     # corruption_type = random.choice([
#     #     "concat", "repeat_adapter_5p",
#     #     "repeat_adapter_3p",
#     # ])

#     corruption_type = random.choice([
#         "concat"
#     ])

#     if corruption_type == "concat":
#         read1, label1 = generate_valid_read(segments_order,
#                                             segments_patterns,
#                                             length_range,
#                                             transcriptome_records)
#         read2, label2 = generate_valid_read(segments_order,
#                                             segments_patterns,
#                                             length_range,
#                                             transcriptome_records)

#         # Randomly reverse or reverse complement second read
#         strand_flip = random.choices(["none", "reverse", "revcomp"],
#                                      weights=[0.5, 0.25, 0.25])[0]
   
#         if strand_flip == "reverse":
#             read2 = read2[::-1]
#             label2 = label2[::-1]
#         if strand_flip == "revcomp":
#             read2 = reverse_complement(read2)
#             label2 = label2[::-1]  # reverse label direction to match RC

#         return read1 + read2, label1 + label2

#     elif corruption_type == "repeat_adapter_5p":
#         adapter_seq, adapter_label = generate_segment(segments_order[1],
#                                                       segments_patterns[1],
#                                                       length_range,
#                                                       transcriptome_records)
#         repeated_adapter = adapter_seq * 3
#         repeated_labels = [adapter_label[0]] * len(repeated_adapter)
#         read, label = generate_valid_read(segments_order,
#                                           segments_patterns,
#                                           length_range,
#                                           transcriptome_records)
#         return repeated_adapter + read, repeated_labels + label

#     elif corruption_type == "repeat_adapter_3p":
#         adapter_seq, adapter_label = generate_segment(segments_order[-2],
#                                                       segments_patterns[-2],
#                                                       length_range,
#                                                       transcriptome_records)
#         repeated_adapter = adapter_seq * 3
#         repeated_labels = [adapter_label[0]] * len(repeated_adapter)
#         read, label = generate_valid_read(segments_order,
#                                           segments_patterns,
#                                           length_range,
#                                           transcriptome_records)
#         return repeated_adapter + read, repeated_labels + label

def generate_invalid_read(segments_order, segments_patterns,
                          length_range, transcriptome_records):
    corruption_type = random.choice([
        "concat", "repeat_adapter_5p",
        "repeat_adapter_3p", "truncate_5p", "truncate_3p"
    ])

    # corruption_type = random.choice([
    #     "concat", "truncate_5p", "truncate_3p"
    # ])

    if corruption_type == "concat":
        read1, label1 = generate_valid_read(
            segments_order,
            segments_patterns,
            length_range,
            transcriptome_records
        )
        read2, label2 = generate_valid_read(
            segments_order,
            segments_patterns,
            length_range,
            transcriptome_records
        )

        # Randomly reverse or reverse complement second read
        strand_flip = random.choices(
            ["none", "reverse", "revcomp"],
            weights=[0.5, 0.25, 0.25]
        )[0]

        if strand_flip == "reverse":
            read2 = read2[::-1]
            label2 = label2[::-1]
        elif strand_flip == "revcomp":
            read2 = reverse_complement(read2)
            # labels follow the reversed orientation
            label2 = label2[::-1]

        return read1 + read2, label1 + label2

    elif corruption_type == "repeat_adapter_5p":
        adapter_seq, adapter_label = generate_segment(
            segments_order[1],
            segments_patterns[1],
            length_range,
            transcriptome_records
        )
        repeated_adapter = adapter_seq * 3
        repeated_labels = [adapter_label[0]] * len(repeated_adapter)

        read, label = generate_valid_read(
            segments_order,
            segments_patterns,
            length_range,
            transcriptome_records
        )
        return repeated_adapter + read, repeated_labels + label

    elif corruption_type == "repeat_adapter_3p":
        adapter_seq, adapter_label = generate_segment(
            segments_order[-2],
            segments_patterns[-2],
            length_range,
            transcriptome_records
        )
        repeated_adapter = adapter_seq * 3
        repeated_labels = [adapter_label[0]] * len(repeated_adapter)

        read, label = generate_valid_read(
            segments_order,
            segments_patterns,
            length_range,
            transcriptome_records
        )
        return read + repeated_adapter, label + repeated_labels

    # ----------------------
    # Truncation artifacts
    # ----------------------
    elif corruption_type in ("truncate_5p", "truncate_3p"):
        read, label = generate_valid_read(
            segments_order,
            segments_patterns,
            length_range,
            transcriptome_records
        )

        read_len = len(read)
        if read_len <= 1:
            # Fallback: if something degenerate happens, just return the read
            return read, label

        # Keep at least some fraction of the read so it's not empty
        min_fraction_kept = 0.3  # you can tune this
        max_trunc = max(1, int(read_len * (1 - min_fraction_kept)))

        # If read is extremely short, avoid truncation turning it invalid for your code
        if max_trunc >= read_len:
            max_trunc = read_len - 1

        trunc_len = random.randint(1, max_trunc)

        if corruption_type == "truncate_5p":
            # Drop bases from the 5' end
            new_read = read[trunc_len:]
            new_label = label[trunc_len:]
        else:  # "truncate_3p"
            # Drop bases from the 3' end
            new_read = read[:-trunc_len]
            new_label = label[:-trunc_len]

        # Safety: in case truncation wiped everything out
        if len(new_read) == 0:
            return read, label

        return new_read, new_label

############## simulate complete batch ##############


def simulate_dynamic_batch_complete(
    num_reads, length_range,
    mismatch_rate, insertion_rate, deletion_rate,
    polyT_error_rate, max_insertions,
    segments_order, segments_patterns,
    transcriptome_records, invalid_fraction, rc
):
    reads, labels = [], []

    for _ in range(num_reads):
        # Generate clean read first
        if np.random.rand() < invalid_fraction:
            sequence, label = generate_invalid_read(
                segments_order, segments_patterns,
                length_range, transcriptome_records
            )
        else:
            sequence, label = generate_valid_read(
                segments_order, segments_patterns,
                length_range, transcriptome_records
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
    transcriptome_records, invalid_fraction
):
    args_complete = (
        num_reads, length_range, mismatch_rate, insertion_rate, deletion_rate,
        polyT_error_rate, max_insertions,
        segments_order, segments_patterns,
        transcriptome_records, invalid_fraction, rc
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
