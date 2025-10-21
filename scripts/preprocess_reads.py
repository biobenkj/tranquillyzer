import os
import gc
import gzip
import logging
import time
import glob
import polars as pl
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor

from filelock import FileLock  # Import the FileLock library

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def determine_bin(length, bin_size=500):
    if length < 10000:
        bin_size = bin_size
    elif length < 50000:
        bin_size = 5000
    elif length < 100000:
        bin_size = 10000
    else:
        bin_size = 25000
    bin_start = (length // bin_size) * bin_size
    bin_end = bin_start + bin_size - 1
    return f"{bin_start}_{bin_end}bp"


def extract_and_bin_reads(file_path, batch_size, 
                          output_dir, output_base_qual):
    reads_by_bin = {}
    file_format = 'fasta' if file_path.endswith(('.fa', '.fasta', '.fa.gz', '.fasta.gz')) else 'fastq'

    with (gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r')) as handle:
        for record in SeqIO.parse(handle, file_format):
            read_length = len(record.seq)
            bin_name = determine_bin(read_length)

            # Ensure that the bin is initialized with all required keys
            if output_base_qual:
                if bin_name not in reads_by_bin:
                    reads_by_bin[bin_name] = {'read_names': [], 'reads': [],
                                              'read_lengths': [], 'base_quals': []}
                reads_by_bin[bin_name]['base_quals'].append(record.format("fastq").splitlines()[3])
            
            else:
                if bin_name not in reads_by_bin:
                    reads_by_bin[bin_name] = {'read_names': [], 'reads': [],
                                              'read_lengths': []}

            reads_by_bin[bin_name]['read_names'].append(record.id)
            reads_by_bin[bin_name]['reads'].append(str(record.seq))
            reads_by_bin[bin_name]['read_lengths'].append(read_length)

            # Once a bin reaches the batch size, save it and reset
            if len(reads_by_bin[bin_name]['reads']) >= batch_size:
                dump_bin_data(output_dir, output_base_qual, bin_name, 
                              reads_by_bin[bin_name])
                if output_base_qual:
                    reads_by_bin[bin_name] = {'read_names': [], 'reads': [],
                                              'read_lengths': [], 'base_quals': []}
                    reads_by_bin[bin_name]['base_quals'].append(record.format("fastq").splitlines()[3])
                else:
                    reads_by_bin[bin_name] = {'read_names': [],
                                              'reads': [],
                                              'read_lengths': []}

        # Dump any remaining data after file is fully read
        for bin_name, data in reads_by_bin.items():
            if data['reads']:
                dump_bin_data(output_dir, output_base_qual, bin_name, data)


def dump_bin_data(output_dir, output_base_qual, bin_name, data):
    os.makedirs(output_dir, exist_ok=True)
    tsv_filename = os.path.join(output_dir, f"{bin_name}.tsv")
    lock_filename = tsv_filename + '.lock'  # Create a lock file for the TSV

    if len(data['reads']) == 0:
        return

    if output_base_qual:
        df = pl.DataFrame({
            "ReadName": data['read_names'],
            "read": data['reads'],
            "read_length": data['read_lengths'],
            "base_qualities": data['base_quals']
        })
    else:
        df = pl.DataFrame({
            "ReadName": data['read_names'],
            "read": data['reads'],
            "read_length": data['read_lengths']
        })

    try:
        # Use a file lock to prevent concurrent writes
        # Ensure only one process writes to this file at a time
        with FileLock(lock_filename):
            write_header = not os.path.exists(tsv_filename) or os.path.getsize(tsv_filename) == 0

            with open(tsv_filename, 'a') as f:
                if write_header:
                    f.write("\t".join(df.columns) + "\n")
                for row in df.to_numpy():
                    f.write("\t".join(map(str, row)) + "\n")

    except Exception as e:
        logger.error(f"Error writing {tsv_filename}: {e}")


def parallel_preprocess_data(file_list, output_dir, batch_size, 
                             output_base_qual, num_workers=4):
    total_files = len(file_list)

    if total_files < num_workers:
        num_workers = total_files
        logger.info(f"Adjusting number of workers to {num_workers} since there are only {total_files} files.")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_path in file_list:
            executor.submit(extract_and_bin_reads, file_path,
                            batch_size, output_dir, output_base_qual)

    end_time = time.time()
    logger.info(f"Processed {total_files} files in {end_time - start_time:.2f} seconds.")

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
            df.sink_parquet(parquet_file,
                            compression="snappy",
                            row_group_size=row_group_size)
            logger.info(f"Converted {tsv_file} to {parquet_file}")

            # Add entries to the read index
            for row in df.collect().to_dict(as_series=False)["ReadName"]:
                read_index[row] = parquet_file

            # Remove the TSV file after conversion
            os.remove(tsv_file)
            logger.info(f"Removed original TSV file: {tsv_file}")
            del df
            gc.collect()

        except Exception as e:
            logger.error(f"Error converting {tsv_file} to Parquet: {e}")

    # Save the index as a Parquet file
    if read_index:
        index_parquet_file = os.path.join(tsv_dir, "read_index.parquet")
        index_df = pl.DataFrame([{"ReadName": k, "ParquetFile": v} for k, v in read_index.items()])
        index_df.write_parquet(index_parquet_file)
        logger.info(f"Index file saved at {index_parquet_file}")


def find_sequence_files(directory):
    extensions = ['*.fa', '*.fasta', '*.fa.gz', '*.fasta.gz',
                  '*.fq', '*.fastq', '*.fq.gz', '*.fastq.gz']
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(directory, ext)))
    return file_list
