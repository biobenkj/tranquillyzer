from Bio import SeqIO
import gzip

def extract_read_ends_w_names(fasta_file, n):
    read_end_names = []
    read_ends = []
    actual_lengths = []
    original_read_end_names = []

    full_length_read_names = []
    full_length_reads = []
    full_length_read_lengths = []
    
    # Check if the file is gzip-compressed
    if fasta_file.endswith('.gz'):
        with gzip.open(fasta_file, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                read_name = record.id
                read_sequence = str(record.seq)
                actual_length = len(read_sequence)

                # Extract n bases from the start and end
                if actual_length >= (2 * n):
                    start_part = read_sequence[:n]
                    end_part = read_sequence[-n:]
                    
                    read_end_names.append(read_name + '_1')
                    read_ends.append(start_part)
                    
                    read_end_names.append(read_name + '_2')
                    read_ends.append(end_part)
                    
                    original_read_end_names.append(read_name)
                    actual_lengths.append(actual_length)

                else:
                    full_length_read_names.append(read_name)
                    full_length_reads.append(read_sequence)
                    full_length_read_lengths.append(actual_length)
                
    else:
        with open(fasta_file, 'r') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                read_name = record.id
                read_sequence = str(record.seq)
                actual_length = len(read_sequence)

                # Extract n bases from the start and end
                if actual_length >= (2 * n):
                    start_part = read_sequence[:n]
                    end_part = read_sequence[-n:]
                    
                    read_end_names.append(read_name + '_1')
                    read_ends.append(start_part)
                    
                    read_end_names.append(read_name + '_2')
                    read_ends.append(end_part)
                    
                    original_read_end_names.append(read_name)
                    actual_lengths.append(actual_length)

                else:
                    full_length_read_names.append(read_name)
                    full_length_reads.append(read_sequence)
                    full_length_read_lengths.append(actual_length)

    return original_read_end_names, read_end_names, read_ends, actual_lengths, full_length_read_names, full_length_reads, full_length_read_lengths

from Bio import SeqIO
import gzip
import multiprocessing
import pickle
import os

def process_record(record, n):
    read_name = record.id
    read_sequence = str(record.seq)
    actual_length = len(read_sequence)

    if actual_length >= (2 * n):
        start_part = read_sequence[:n]
        end_part = read_sequence[-n:]
        return (read_name, read_name + '_1', start_part, read_name + '_2', end_part, actual_length, None, None, None)
    else:
        return (read_name, None, None, None, None, None, read_name, read_sequence, actual_length)

def write_chunk_to_files(results_chunk, output_dir, chunk_index):
    read_ends_dir = os.path.join(output_dir, "read_ends_pp_fa")
    if not os.path.exists(read_ends_dir):
        os.makedirs(read_ends_dir)

    # Write to separate .pkl files
    with open(os.path.join(read_ends_dir, f"read_ends_{chunk_index}.pkl"), 'wb') as f:
        pickle.dump(results_chunk[0], f)
    with open(os.path.join(read_ends_dir, f"read_end_names_{chunk_index}.pkl"), 'wb') as f:
        pickle.dump(results_chunk[1], f)
    with open(os.path.join(read_ends_dir, f"original_read_end_names_{chunk_index}.pkl"), 'wb') as f:
        pickle.dump(results_chunk[2], f)
    with open(os.path.join(read_ends_dir, f"actual_lengths_{chunk_index}.pkl"), 'wb') as f:
        pickle.dump(results_chunk[3], f)
    with open(os.path.join(read_ends_dir, f"full_length_read_names_{chunk_index}.pkl"), 'wb') as f:
        pickle.dump(results_chunk[4], f)
    with open(os.path.join(read_ends_dir, f"full_length_reads_{chunk_index}.pkl"), 'wb') as f:
        pickle.dump(results_chunk[5], f)
    with open(os.path.join(read_ends_dir, f"full_length_read_lengths_{chunk_index}.pkl"), 'wb') as f:
        pickle.dump(results_chunk[6], f)

def combine_pickle_files(output_dir, file_pattern, output_file):
    combined_data = []

    chunk_files = [f for f in os.listdir(output_dir) if f.startswith(file_pattern)]
    chunk_files.sort()

    for chunk_file in chunk_files:
        chunk_file_path = os.path.join(output_dir, chunk_file)
        with open(chunk_file_path, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    combined_data.extend(data)
                except EOFError:
                    break

    with open(os.path.join(output_dir, output_file), 'wb') as f:
        pickle.dump(combined_data, f)

def combine_all_chunks(output_dir):
    read_ends_dir = os.path.join(output_dir, "read_ends_pp_fa")
    combine_pickle_files(read_ends_dir, "read_ends_", "read_ends.pkl")
    combine_pickle_files(read_ends_dir, "read_end_names_", "read_end_names.pkl")
    combine_pickle_files(read_ends_dir, "original_read_end_names_", "original_read_end_names.pkl")
    combine_pickle_files(read_ends_dir, "actual_lengths_", "actual_lengths.pkl")
    combine_pickle_files(read_ends_dir, "full_length_read_names_", "full_length_read_names.pkl")
    combine_pickle_files(read_ends_dir, "full_length_reads_", "full_length_reads.pkl")
    combine_pickle_files(read_ends_dir, "full_length_read_lengths_", "full_length_read_lengths.pkl")

def extract_read_ends_w_names(file_path, n, output_dir=None, num_workers=None, chunk_size=1000):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    open_func = gzip.open if file_path.endswith('.gz') else open
    file_format = file_path.split('.')[-2 if file_path.endswith('.gz') else -1].lower()

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    with open_func(file_path, 'rt') as handle:
        if file_format == 'fasta':
            iterator = SeqIO.FastaIterator(handle)
        elif file_format == 'fastq':
            iterator = SeqIO.parse(handle, 'fastq')
        else:
            raise ValueError("Unsupported file format. Supported formats are: FASTA (.fasta, .fasta.gz) and FASTQ (.fastq, .fastq.gz)")

        total_records = 0
        chunk_index = 0
        progress_interval = 1000
        pool = multiprocessing.Pool(processes=num_workers)

        results_chunk = [[], [], [], [], [], [], []]

        while True:
            records = []
            try:
                for _ in range(chunk_size):
                    records.append(next(iterator))
                    total_records += 1
            except StopIteration:
                break

            results = pool.starmap(process_record, [(record, n) for record in records])

            for result in results:
                if result[1] is not None:
                    results_chunk[0].append(result[1])
                    results_chunk[1].append(result[2])
                    results_chunk[2].append(result[0])
                    results_chunk[3].append(result[5])
                else:
                    results_chunk[4].append(result[6])
                    results_chunk[5].append(result[7])
                    results_chunk[6].append(result[8])

            if len(results_chunk[0]) >= chunk_size:
                write_chunk_to_files(results_chunk, output_dir, chunk_index)
                chunk_index += 1
                results_chunk = [[], [], [], [], [], [], []]

            if total_records % progress_interval == 0:
                print(f"Processed {total_records} records.")

        if results_chunk[0]:
            write_chunk_to_files(results_chunk, output_dir, chunk_index)

    pool.close()
    pool.join()

    # Combine chunk files into single files
    combine_all_chunks(output_dir)

