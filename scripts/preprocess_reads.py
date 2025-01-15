# import os
# import gzip
# import pickle
# import csv
# import glob
# from Bio import SeqIO
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import Manager, Value, Lock
# import time
# import logging
# import math

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# def determine_bin_sizes(max_length, bin_size=500):
#     bins = []
#     for start in range(0, max_length + bin_size, bin_size):
#         end = start + bin_size - 1
#         bins.append((start, end))
#     return bins

# def get_bin_name(length, bins):
#     for bin_start, bin_end in bins:
#         if bin_start <= length <= bin_end:
#             return f"{bin_start}_{bin_end}bp"
#     return None

# def extract_reads_and_names(file_path, batch_size, bins, resume_position=None):
#     reads_by_bin = {f"{start}_{end}bp": {'reads': [], 'read_names': [], 'read_lengths': []}
#                     for start, end in bins}

#     file_format = 'fasta' if file_path.endswith(('.fa', '.fasta', '.fa.gz', '.fasta.gz')) else 'fastq'

#     if file_path.endswith('.gz'):
#         handle = gzip.open(file_path, 'rt')
#     else:
#         handle = open(file_path, 'r')

#     if resume_position:
#         handle.seek(resume_position)

#     with handle:
#         for record in SeqIO.parse(handle, file_format):
#             read_name = record.id
#             read_sequence = str(record.seq)
#             read_length = len(read_sequence)

#             bin_name = get_bin_name(read_length, bins)
#             if bin_name:
#                 reads_by_bin[bin_name]['reads'].append(read_sequence)
#                 reads_by_bin[bin_name]['read_names'].append(read_name)
#                 reads_by_bin[bin_name]['read_lengths'].append(read_length)

#             if all(len(data['reads']) >= batch_size for data in reads_by_bin.values()):
#                 current_position = handle.tell()
#                 yield reads_by_bin, current_position
#                 # Reset the bins for next batch
#                 reads_by_bin = {f"{start}_{end}bp": {'reads': [], 'read_names': [], 'read_lengths': []}
#                                 for start, end in bins}

#         if any(data['reads'] for data in reads_by_bin.values()):
#             current_position = handle.tell()
#             yield reads_by_bin, current_position

# def save_read_index(output_dir, file_prefix, bin_name, read_names, read_lengths):
#     index_file = os.path.join(output_dir, "read_index.tsv")
#     index_rows = []
#     filename = os.path.join(file_prefix, f"{bin_name}.pkl")
#     for idx, (read_name, read_length) in enumerate(zip(read_names, read_lengths)):
#         index_rows.append([read_name, filename, idx, read_length])

#     with open(index_file, 'a', newline='') as f:
#         writer = csv.writer(f, delimiter='\t')
#         writer.writerows(index_rows)

# def save_bin_as_pkl(data, output_dir, file_prefix, bin_name):
#     output_subdir = os.path.join(output_dir, file_prefix)
#     os.makedirs(output_subdir, exist_ok=True)

#     filename = os.path.join(output_subdir, f"{bin_name}.pkl")
#     with open(filename, 'wb') as f:
#         pickle.dump(data, f)
#     logger.info(f"Saved {bin_name} to {filename}")

#     # Save index information
#     save_read_index(output_dir, file_prefix, bin_name, data['read_names'], data['read_lengths'])

# def preprocess_data(file_path, output_dir, batch_size, chunk_counter, lock):
#     os.makedirs(output_dir, exist_ok=True)

#     # Remove extensions from file_prefix
#     file_prefix = os.path.basename(file_path)
#     for ext in ['.fa', '.fasta', '.fq', '.fastq', '.gz']:
#         if file_prefix.endswith(ext):
#             file_prefix = file_prefix[:-len(ext)]

#     # Determine the longest read length in the file
#     max_read_length = 0
#     with gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r') as handle:
#         for record in SeqIO.parse(handle, 'fasta' if file_path.endswith(('.fa', '.fasta', '.fa.gz', '.fasta.gz')) else 'fastq'):
#             max_read_length = max(max_read_length, len(record.seq))

#     # Dynamically generate bin sizes based on the longest read
#     bins = determine_bin_sizes(max_read_length)

#     gen = extract_reads_and_names(file_path, batch_size, bins)

#     resume_position = None

#     while True:
#         try:
#             reads_by_bin, resume_position = next(gen)
#         except StopIteration:
#             break

#         for bin_name, data in reads_by_bin.items():
#             if data['reads']:
#                 save_bin_as_pkl(data, output_dir, file_prefix, bin_name)

# def find_sequence_files(directory):
#     extensions = ['*.fa', '*.fasta', '*.fa.gz', '*.fasta.gz', '*.fq', '*.fastq', '*.fq.gz', '*.fastq.gz']
#     file_names = []

#     for ext in extensions:
#         file_names.extend(glob.glob(os.path.join(directory, ext)))

#     return file_names

# def parallel_preprocess_data(directory, output_dir, batch_size, num_workers):
#     start_time = time.time()
#     sequence_files = find_sequence_files(directory)

#     manager = Manager()
#     chunk_counter = manager.Value('i', 0)
#     lock = manager.Lock()

#     # Create an empty index file
#     index_file_path = os.path.join(output_dir, "read_index.tsv")
#     with open(index_file_path, 'w', newline='') as index_file:
#         writer = csv.writer(index_file, delimiter='\t')
#         writer.writerow(['ReadName', 'PKL_File', 'Index', 'read_length'])  # Header for the index file

#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         futures = []
#         for file in sequence_files:
#             future = executor.submit(preprocess_data, file, output_dir, batch_size, chunk_counter, lock)
#             futures.append(future)

#         for future in as_completed(futures):
#             try:
#                 future.result()
#             except Exception as e:
#                 logger.error(f"Error processing file: {e}")

#     end_time = time.time()
#     logger.info(f"Processing completed in {end_time - start_time:.2f} seconds.")

# import os
# import gzip
# import polars as pl
# from Bio import SeqIO
# from concurrent.futures import ProcessPoolExecutor
# import logging
# import time
# import glob

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# def determine_bin(length, bin_size=500):
#     bin_start = (length // bin_size) * bin_size
#     bin_end = bin_start + bin_size - 1
#     return f"{bin_start}_{bin_end}bp"

# def extract_and_bin_reads(file_path, batch_size, output_dir):
#     logger.info(f"Processing file: {file_path}")
    
#     reads_by_bin = {}
#     file_format = 'fasta' if file_path.endswith(('.fa', '.fasta', '.fa.gz', '.fasta.gz')) else 'fastq'
    
#     with (gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r')) as handle:
#         for record in SeqIO.parse(handle, file_format):
#             read_length = len(record.seq)
#             bin_name = determine_bin(read_length)

#             if bin_name not in reads_by_bin:
#                 reads_by_bin[bin_name] = {'read_names': [], 'reads': [], 'read_lengths': []}

#             reads_by_bin[bin_name]['read_names'].append(record.id)
#             reads_by_bin[bin_name]['reads'].append(str(record.seq))
#             reads_by_bin[bin_name]['read_lengths'].append(read_length)

#             # Once a bin reaches the batch size, save it and reset
#             if len(reads_by_bin[bin_name]['reads']) >= batch_size:
#                 logger.info(f"Batch size reached for bin {bin_name} in {file_path}. Dumping to file.")
#                 dump_bin_data(output_dir, bin_name, reads_by_bin[bin_name])
#                 reads_by_bin[bin_name] = {'read_names': [], 'reads': [], 'read_lengths': []}

#         # Dump any remaining data after file is fully read
#         for bin_name, data in reads_by_bin.items():
#             if data['reads']:
#                 logger.info(f"Dumping remaining data for bin {bin_name} in {file_path}.")
#                 dump_bin_data(output_dir, bin_name, data)

# def dump_bin_data(output_dir, bin_name, data):
#     os.makedirs(output_dir, exist_ok=True)
#     tsv_filename = os.path.join(output_dir, f"{bin_name}.tsv")

#     logger.info(f"Preparing to save bin {bin_name} to {tsv_filename}. Number of reads: {len(data['reads'])}")

#     if len(data['reads']) == 0:
#         logger.warning(f"No data to save for bin {bin_name}. Skipping.")
#         return

#     df = pl.DataFrame({
#         "ReadName": data['read_names'],
#         "Read": data['reads'],
#         "ReadLength": data['read_lengths']
#     })

#     logger.info(f"DataFrame for bin {bin_name} has {len(df)} rows.")

#     try:
#         # Write the DataFrame as a CSV with tabs as separators
#         with open(tsv_filename, 'a') as f:
#             if os.path.getsize(tsv_filename) == 0:
#                 f.write("\t".join(df.columns) + "\n")  # Write header if file is empty
#             for row in df.to_numpy():
#                 f.write("\t".join(map(str, row)) + "\n")
#         logger.info(f"Successfully written {len(df)} reads to {tsv_filename}.")
#     except Exception as e:
#         logger.error(f"Error writing {tsv_filename}: {e}")

# def parallel_preprocess_data(file_list, output_dir, batch_size, num_workers=4):
#     start_time = time.time()

#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         for file_path in file_list:
#             executor.submit(extract_and_bin_reads, file_path, batch_size, output_dir)

#     end_time = time.time()
#     logger.info(f"All files processed in {end_time - start_time:.2f} seconds.")

# def find_sequence_files(directory):
#     extensions = ['*.fa', '*.fasta', '*.fa.gz', '*.fasta.gz', '*.fq', '*.fastq', '*.fq.gz', '*.fastq.gz']
#     file_list = []
#     for ext in extensions:
#         file_list.extend(glob.glob(os.path.join(directory, ext)))
#     return file_list

# import os
# import gzip
# import polars as pl
# from Bio import SeqIO
# from concurrent.futures import ProcessPoolExecutor
# import logging
# import time
# import glob

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# def determine_bin(length, bin_size=500):
#     bin_start = (length // bin_size) * bin_size
#     bin_end = bin_start + bin_size - 1
#     return f"{bin_start}_{bin_end}bp"

# def extract_and_bin_reads(file_path, batch_size, output_dir):
#     reads_by_bin = {}
#     file_format = 'fasta' if file_path.endswith(('.fa', '.fasta', '.fa.gz', '.fasta.gz')) else 'fastq'
    
#     with (gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r')) as handle:
#         for record in SeqIO.parse(handle, file_format):
#             read_length = len(record.seq)
#             bin_name = determine_bin(read_length)

#             if bin_name not in reads_by_bin:
#                 reads_by_bin[bin_name] = {'read_names': [], 'reads': [], 'read_lengths': []}

#             reads_by_bin[bin_name]['read_names'].append(record.id)
#             reads_by_bin[bin_name]['reads'].append(str(record.seq))
#             reads_by_bin[bin_name]['read_lengths'].append(read_length)

#             # Once a bin reaches the batch size, save it and reset
#             if len(reads_by_bin[bin_name]['reads']) >= batch_size:
#                 dump_bin_data(output_dir, bin_name, reads_by_bin[bin_name])
#                 reads_by_bin[bin_name] = {'read_names': [], 'reads': [], 'read_lengths': []}

#         # Dump any remaining data after file is fully read
#         for bin_name, data in reads_by_bin.items():
#             if data['reads']:
#                 dump_bin_data(output_dir, bin_name, data)

# def dump_bin_data(output_dir, bin_name, data):
#     os.makedirs(output_dir, exist_ok=True)
#     tsv_filename = os.path.join(output_dir, f"{bin_name}.tsv")

#     if len(data['reads']) == 0:
#         return

#     df = pl.DataFrame({
#         "ReadName": data['read_names'],
#         "Read": data['reads'],
#         "ReadLength": data['read_lengths']
#     })

#     try:
#         with open(tsv_filename, 'a') as f:
#             if os.path.getsize(tsv_filename) == 0:
#                 f.write("\t".join(df.columns) + "\n")  # Write header if file is empty
#             for row in df.to_numpy():
#                 f.write("\t".join(map(str, row)) + "\n")
#     except Exception as e:
#         logger.error(f"Error writing {tsv_filename}: {e}")

# def parallel_preprocess_data(file_list, output_dir, batch_size, num_workers=4):
#     start_time = time.time()
#     total_files = len(file_list)

#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         for file_path in file_list:
#             executor.submit(extract_and_bin_reads, file_path, batch_size, output_dir)

#     end_time = time.time()
#     logger.info(f"Processed {total_files} files in {end_time - start_time:.2f} seconds.")

#     # Convert TSV files to Parquet files after all files are processed
#     convert_tsv_to_parquet(output_dir, chunk_size=1000000)

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# def convert_tsv_to_parquet(tsv_dir, chunk_size=100_000):
#     logger.info("Converting TSV files to Parquet files...")
#     os.makedirs(tsv_dir, exist_ok=True)
    
#     # Find all TSV files in the directory
#     tsv_files = glob.glob(os.path.join(tsv_dir, "*.tsv"))
    
#     # Group TSV files by bin name
#     bins = {}
#     for tsv_file in tsv_files:
#         bin_name = os.path.basename(tsv_file).split(".")[0]  # Extract bin name from filename
#         if bin_name not in bins:
#             bins[bin_name] = []
#         bins[bin_name].append(tsv_file)
    
#     for bin_name, files in bins.items():
#         logger.info(f"Processing bin: {bin_name} with {len(files)} files.")
#         temp_parquet_files = []

#         for file in files:
#             logger.info(f"Reading TSV file: {file} in chunks.")
#             try:
#                 # Manually read file in chunks
#                 with open(file, 'r') as f:
#                     # Read the header
#                     header = f.readline().strip().split('\t')
#                     chunk_data = []
#                     row_count = 0
#                     part_index = 0

#                     for line in f:
#                         row = line.strip().split('\t')
#                         chunk_data.append(row)
#                         row_count += 1

#                         # If chunk size is reached, process the chunk
#                         if row_count >= chunk_size:
#                             df = pl.DataFrame(chunk_data, schema=header)  # Create DataFrame with schema
#                             temp_parquet_file = os.path.join(tsv_dir, f"{bin_name}_part{part_index}.parquet")
#                             df.write_parquet(temp_parquet_file)
#                             temp_parquet_files.append(temp_parquet_file)
#                             logger.info(f"Written chunk to {temp_parquet_file}")

#                             # Reset for the next chunk
#                             chunk_data = []
#                             row_count = 0
#                             part_index += 1

#                     # Process any remaining rows as the last chunk
#                     if chunk_data:
#                         df = pl.DataFrame(chunk_data, schema=header)  # Create DataFrame with schema
#                         temp_parquet_file = os.path.join(tsv_dir, f"{bin_name}_part{part_index}.parquet")
#                         df.write_parquet(temp_parquet_file)
#                         temp_parquet_files.append(temp_parquet_file)
#                         logger.info(f"Written final chunk to {temp_parquet_file}")

#             except Exception as e:
#                 logger.error(f"Error reading {file}: {e}")
#                 continue

#             # After successful processing, remove the TSV file
#             try:
#                 os.remove(file)
#                 logger.info(f"Removed original TSV file: {file}")
#             except Exception as e:
#                 logger.error(f"Error removing file {file}: {e}")

#         # Merge all temporary Parquet files into one
#         if temp_parquet_files:
#             logger.info(f"Merging Parquet files for bin {bin_name}.")
#             try:
#                 df_list = [pl.read_parquet(parquet_file) for parquet_file in temp_parquet_files]
#                 combined_df = pl.concat(df_list)
#                 final_parquet_file = os.path.join(tsv_dir, f"{bin_name}.parquet")
#                 combined_df.write_parquet(final_parquet_file)
#                 logger.info(f"Successfully merged and written Parquet file for bin {bin_name}: {final_parquet_file}")
#             except Exception as e:
#                 logger.error(f"Error merging Parquet files for bin {bin_name}: {e}")
#                 continue

#             # Remove the temporary Parquet files
#             for temp_file in temp_parquet_files:
#                 try:
#                     os.remove(temp_file)
#                     logger.info(f"Removed temporary Parquet file: {temp_file}")
#                 except Exception as e:
#                     logger.error(f"Error removing temporary Parquet file {temp_file}: {e}")

# def find_sequence_files(directory):
#     extensions = ['*.fa', '*.fasta', '*.fa.gz', '*.fasta.gz', '*.fq', '*.fastq', '*.fq.gz', '*.fastq.gz']
#     file_list = []
#     for ext in extensions:
#         file_list.extend(glob.glob(os.path.join(directory, ext)))
#     return file_list

# import os
# import gzip
# import polars as pl
# from Bio import SeqIO
# from concurrent.futures import ProcessPoolExecutor
# import logging
# import time
# import glob

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# def determine_bin(length, bin_size=500):
#     bin_start = (length // bin_size) * bin_size
#     bin_end = bin_start + bin_size - 1
#     return f"{bin_start}_{bin_end}bp"

# def extract_and_bin_reads(file_path, batch_size, output_dir):
#     reads_by_bin = {}
#     file_format = 'fasta' if file_path.endswith(('.fa', '.fasta', '.fa.gz', '.fasta.gz')) else 'fastq'
    
#     with (gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r')) as handle:
#         for record in SeqIO.parse(handle, file_format):
#             read_length = len(record.seq)
#             bin_name = determine_bin(read_length)

#             if bin_name not in reads_by_bin:
#                 reads_by_bin[bin_name] = {'read_names': [], 'reads': [], 'read_lengths': []}

#             reads_by_bin[bin_name]['read_names'].append(record.id)
#             reads_by_bin[bin_name]['reads'].append(str(record.seq))
#             reads_by_bin[bin_name]['read_lengths'].append(read_length)

#             # Once a bin reaches the batch size, save it and reset
#             if len(reads_by_bin[bin_name]['reads']) >= batch_size:
#                 dump_bin_data(output_dir, bin_name, reads_by_bin[bin_name])
#                 reads_by_bin[bin_name] = {'read_names': [], 'reads': [], 'read_lengths': []}

#         # Dump any remaining data after file is fully read
#         for bin_name, data in reads_by_bin.items():
#             if data['reads']:
#                 dump_bin_data(output_dir, bin_name, data)

# def dump_bin_data(output_dir, bin_name, data):
#     os.makedirs(output_dir, exist_ok=True)
#     tsv_filename = os.path.join(output_dir, f"{bin_name}.tsv")

#     if len(data['reads']) == 0:
#         return

#     df = pl.DataFrame({
#         "ReadName": data['read_names'],
#         "Read": data['reads'],
#         "ReadLength": data['read_lengths']
#     })

#     try:
#         with open(tsv_filename, 'a') as f:
#             if os.path.getsize(tsv_filename) == 0:
#                 f.write("\t".join(df.columns) + "\n")  # Write header if file is empty
#             for row in df.to_numpy():
#                 f.write("\t".join(map(str, row)) + "\n")
#     except Exception as e:
#         logger.error(f"Error writing {tsv_filename}: {e}")

# def parallel_preprocess_data(file_list, output_dir, batch_size, num_workers=4):
#     start_time = time.time()
#     total_files = len(file_list)

#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         for file_path in file_list:
#             executor.submit(extract_and_bin_reads, file_path, batch_size, output_dir)

#     end_time = time.time()
#     logger.info(f"Processed {total_files} files in {end_time - start_time:.2f} seconds.")

#     # Convert TSV files to chunked Parquet files after all files are processed
#     convert_tsv_to_parquet(output_dir, chunk_size=1000000)

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# def convert_tsv_to_parquet(tsv_dir, chunk_size=100_000):
#     logger.info("Converting TSV files to Parquet files...")
#     os.makedirs(tsv_dir, exist_ok=True)
    
#     # Find all TSV files in the directory
#     tsv_files = glob.glob(os.path.join(tsv_dir, "*.tsv"))
#     index_data = []

#     for tsv_file in tsv_files:
#         bin_name = os.path.basename(tsv_file).split(".")[0]  # Extract bin name from filename
#         logger.info(f"Processing bin: {bin_name}")

#         try:
#             # Read the TSV file using Polars with explicit schema and correct column mapping
#             df = pl.read_csv(
#                 tsv_file, 
#                 separator='\t',  # Use 'separator' for specifying tab separation
#                 columns=["ReadName", "Read", "ReadLength"],  # Ensure the correct order of columns
#                 dtypes={
#                     "ReadName": pl.Utf8,
#                     "Read": pl.Utf8,
#                     "ReadLength": pl.Int64  # Explicitly set ReadLength as Int64
#                 }
#             )

#             # Iterate through the DataFrame in chunks and save as Parquet files
#             for i in range(0, len(df), chunk_size):
#                 chunk = df.slice(i, chunk_size)
#                 chunk_parquet_file = os.path.join(tsv_dir, f"{bin_name}_part{i//chunk_size}.parquet")
#                 chunk.write_parquet(chunk_parquet_file)
                
#                 # Add entries to the index
#                 index_data.extend([(row[0], chunk_parquet_file) for row in chunk.rows()])

#                 logger.info(f"Written chunk to {chunk_parquet_file}")

#         except Exception as e:
#             logger.error(f"Error reading {tsv_file}: {e}")
#             continue

#         # Remove the TSV file after processing
#         try:
#             os.remove(tsv_file)
#             logger.info(f"Removed original TSV file: {tsv_file}")
#         except Exception as e:
#             logger.error(f"Error removing file {tsv_file}: {e}")

#     # Save the index as a Parquet file
#     if index_data:
#         index_df = pl.DataFrame(index_data, schema=["ReadName", "ParquetFile"])
#         index_parquet_file = os.path.join(tsv_dir, "read_index.parquet")
#         index_df.write_parquet(index_parquet_file)
#         logger.info(f"Index file saved at {index_parquet_file}")

# def process_chunk(chunk_data, header, bin_name, part_index, tsv_dir, index_data):
#     try:
#         df = pl.DataFrame(chunk_data, schema=header)  # Create DataFrame with schema

#         # Ensure the 'ReadLength' column is correctly typed as integer
#         df = df.with_columns(pl.col("ReadLength").cast(pl.Int64, strict=False))

#         chunk_parquet_file = os.path.join(tsv_dir, f"{bin_name}_part{part_index}.parquet")
#         df.write_parquet(chunk_parquet_file)

#         # Add entries to the index
#         index_data.extend([(record[0], chunk_parquet_file) for record in chunk_data])

#         logger.info(f"Written chunk to {chunk_parquet_file}")
#     except Exception as e:
#         logger.error(f"Error converting chunk for bin {bin_name} part {part_index}: {e}")

# def find_sequence_files(directory):
#     extensions = ['*.fa', '*.fasta', '*.fa.gz', '*.fasta.gz', '*.fq', '*.fastq', '*.fq.gz', '*.fastq.gz']
#     file_list = []
#     for ext in extensions:
#         file_list.extend(glob.glob(os.path.join(directory, ext)))
#     return file_list

import os
import gzip
import polars as pl
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor
import logging
import time
import glob
from filelock import FileLock  # Import the FileLock library

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def determine_bin(length, bin_size=500):
    bin_start = (length // bin_size) * bin_size
    bin_end = bin_start + bin_size - 1
    return f"{bin_start}_{bin_end}bp"

def extract_and_bin_reads(file_path, batch_size, output_dir):
    reads_by_bin = {}
    file_format = 'fasta' if file_path.endswith(('.fa', '.fasta', '.fa.gz', '.fasta.gz')) else 'fastq'
    
    with (gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r')) as handle:
        for record in SeqIO.parse(handle, file_format):
            read_length = len(record.seq)
            bin_name = determine_bin(read_length)

            # Ensure that the bin is initialized with all required keys
            if bin_name not in reads_by_bin:
                reads_by_bin[bin_name] = {'read_names': [], 'reads': [], 'read_lengths': []}

            reads_by_bin[bin_name]['read_names'].append(record.id)
            reads_by_bin[bin_name]['reads'].append(str(record.seq))
            reads_by_bin[bin_name]['read_lengths'].append(read_length)

            # Once a bin reaches the batch size, save it and reset
            if len(reads_by_bin[bin_name]['reads']) >= batch_size:
                dump_bin_data(output_dir, bin_name, reads_by_bin[bin_name])
                reads_by_bin[bin_name] = {'read_names': [], 'reads': [], 'read_lengths': []}

        # Dump any remaining data after file is fully read
        for bin_name, data in reads_by_bin.items():
            if data['reads']:
                dump_bin_data(output_dir, bin_name, data)

def dump_bin_data(output_dir, bin_name, data):
    os.makedirs(output_dir, exist_ok=True)
    tsv_filename = os.path.join(output_dir, f"{bin_name}.tsv")
    lock_filename = tsv_filename + '.lock'  # Create a lock file for the TSV

    if len(data['reads']) == 0:
        return

    df = pl.DataFrame({
        "ReadName": data['read_names'],
        "read": data['reads'],
        "read_length": data['read_lengths']
    })

    try:
        # Use a file lock to prevent concurrent writes
        with FileLock(lock_filename):  # Ensure only one process writes to this file at a time
            write_header = not os.path.exists(tsv_filename) or os.path.getsize(tsv_filename) == 0

            with open(tsv_filename, 'a') as f:
                if write_header:
                    f.write("\t".join(df.columns) + "\n")  # Write header if the file is empty
                for row in df.to_numpy():
                    f.write("\t".join(map(str, row)) + "\n")

    except Exception as e:
        logger.error(f"Error writing {tsv_filename}: {e}")

def parallel_preprocess_data(file_list, output_dir, batch_size, num_workers=4):
    total_files = len(file_list)

    # Dynamically adjust the number of workers to the number of files, if fewer files than threads
    if total_files < num_workers:
        num_workers = total_files
        logger.info(f"Adjusting number of workers to {num_workers} since there are only {total_files} files.")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_path in file_list:
            executor.submit(extract_and_bin_reads, file_path, batch_size, output_dir)

    end_time = time.time()
    logger.info(f"Processed {total_files} files in {end_time - start_time:.2f} seconds.")

    # Convert TSV files to chunked Parquet files after all files are processed
    os.system("rm " + output_dir + "/*.lock")
    convert_tsv_to_parquet(output_dir, row_group_size=1000000)

def convert_tsv_to_parquet(tsv_dir, row_group_size=1000000):
    logger.info("Converting TSV files to Parquet files...")
    os.makedirs(tsv_dir, exist_ok=True)
    
    # Find all TSV files in the directory
    tsv_files = glob.glob(os.path.join(tsv_dir, "*.tsv"))
    read_index = {}

    for tsv_file in tsv_files:
        bin_name = os.path.basename(tsv_file).split(".")[0]
        try:
            # Use scan_csv to lazily read the TSV and sink it to Parquet
            df = pl.scan_csv(tsv_file, separator='\t')
            parquet_file = os.path.join(tsv_dir, f"{bin_name}.parquet")

            logger.info(f"\nConverting {tsv_file}")
            df.sink_parquet(parquet_file, compression="snappy", row_group_size=row_group_size)
            logger.info(f"Converted {tsv_file} to {parquet_file}")

            # Add entries to the read index
            for row in df.collect().to_dict(as_series=False)["ReadName"]:
                read_index[row] = parquet_file

            # Remove the TSV file after conversion
            os.remove(tsv_file)
            logger.info(f"Removed original TSV file: {tsv_file}")

        except Exception as e:
            logger.error(f"Error converting {tsv_file} to Parquet: {e}")

    # Save the index as a Parquet file
    if read_index:
        index_parquet_file = os.path.join(tsv_dir, "read_index.parquet")
        index_df = pl.DataFrame([{"ReadName": k, "ParquetFile": v} for k, v in read_index.items()])
        index_df.write_parquet(index_parquet_file)
        logger.info(f"Index file saved at {index_parquet_file}")

def find_sequence_files(directory):
    extensions = ['*.fa', '*.fasta', '*.fa.gz', '*.fasta.gz', '*.fq', '*.fastq', '*.fq.gz', '*.fastq.gz']
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(directory, ext)))
    return file_list