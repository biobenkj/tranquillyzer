# from Bio import SeqIO
# import gzip

# def extract_full_reads_w_names(fasta_file):
#     reads = []
#     read_names = []
#     lengths = []

#     # Check if the file is gzip-compressed
#     if fasta_file.endswith('.gz'):
#         with gzip.open(fasta_file, 'rt') as handle:
#             for record in SeqIO.parse(handle, 'fasta'):
#                 read_names.append(record.id)
#                 reads.append(str(record.seq))
#                 lengths.append(len(record.seq))
#     else:
#         with open(fasta_file, 'r') as handle:
#             for record in SeqIO.parse(handle, 'fasta'):
#                 read_names.append(record.id)
#                 reads.append(str(record.seq))
#                 lengths.append(len(record.seq))

#     return read_names, reads, lengths

from Bio import SeqIO
import gzip
import multiprocessing
import pickle
import os

def process_record(record):
    return record.id, str(record.seq), len(record.seq)

def extract_full_reads_w_names(file_path, output_dir=None, num_workers=None, chunk_size=1000):
    open_func = gzip.open if file_path.endswith('.gz') else open
    file_format = file_path.split('.')[-2 if file_path.endswith('.gz') else -1].lower()

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def write_to_files(read_names, reads, lengths, output_dir, chunk_index):
        full_length_dir = os.path.join(output_dir, "full_length_pp_fa")
        if not os.path.exists(full_length_dir):
            os.makedirs(full_length_dir)
        
        with open(os.path.join(full_length_dir, f"read_names_{chunk_index}.pkl"), 'wb') as f:
            pickle.dump(read_names, f)
        with open(os.path.join(full_length_dir, f"reads_{chunk_index}.pkl"), 'wb') as f:
            pickle.dump(reads, f)
        with open(os.path.join(full_length_dir, f"lengths_{chunk_index}.pkl"), 'wb') as f:
            pickle.dump(lengths, f)

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
        
        read_names, reads, lengths = [], [], []

        while True:
            records = []
            try:
                for _ in range(chunk_size):
                    records.append(next(iterator))
                    total_records += 1
            except StopIteration:
                break

            results = pool.map(process_record, records)

            for result in results:
                read_names.append(result[0])
                reads.append(result[1])
                lengths.append(result[2])

            if output_dir:
                write_to_files(read_names, reads, lengths, output_dir, chunk_index)
                chunk_index += 1
                read_names, reads, lengths = [], [], []

            if total_records % progress_interval == 0:
                print(f"Processed {total_records} reads.")

        if read_names:
            write_to_files(read_names, reads, lengths, output_dir, chunk_index)

    pool.close()
    pool.join()

    combine_all_full_len_chunks(output_dir)

def combine_full_len_pickle_files(output_dir, file_pattern, output_file):
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

def combine_all_full_len_chunks(output_dir):
    full_length_dir = os.path.join(output_dir, "full_length_pp_fa")
    combine_full_len_pickle_files(full_length_dir, "read_names_", "read_names.pkl")
    combine_full_len_pickle_files(full_length_dir, "reads_", "reads.pkl")
    combine_full_len_pickle_files(full_length_dir, "lengths_", "lengths.pkl")

