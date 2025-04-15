import os
import gc
import pickle
import logging
import numpy as np
import polars as pl
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from numba import njit

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a NumPy lookup array (256 elements to handle all ASCII characters)
NUCLEOTIDE_TO_ID = np.zeros(256, dtype=np.int8)
NUCLEOTIDE_TO_ID[ord('A')] = 1
NUCLEOTIDE_TO_ID[ord('C')] = 2
NUCLEOTIDE_TO_ID[ord('G')] = 3
NUCLEOTIDE_TO_ID[ord('T')] = 4
NUCLEOTIDE_TO_ID[ord('N')] = 5  # Default encoding for unknown nucleotides

# Enable memory growth to avoid pre-allocating all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def encode_sequence(read):
    nucleotide_to_id = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}
    return [nucleotide_to_id[n] for n in read]

def encode_sequences_parallel(sequences, num_workers=64):
    """Parallelized encoding of sequences using multiprocessing."""
    with mp.Pool(num_workers) as pool:
        encoded_sequences = pool.map(encode_sequence, sequences)
    return encoded_sequences

tf.config.optimizer.set_jit(True)

@njit
def encode_sequence_numba(read):
    """Fast Numba-based nucleotide encoding using ASCII lookup."""
    encoded_seq = np.zeros(len(read), dtype=np.int8)
    for i in range(len(read)):
        encoded_seq[i] = NUCLEOTIDE_TO_ID[ord(read[i])]  # Faster lookup using ASCII index
    return encoded_seq

def preprocess_sequences(sequences):
    """Converts DNA sequences into NumPy integer arrays for Numba processing."""
    max_len = max(len(seq) for seq in sequences)  # Get max sequence length
    encoded_array = np.zeros((len(sequences), max_len), dtype=np.int8)  # Pre-allocate array

    for i, seq in enumerate(sequences):
        encoded_array[i, :len(seq)] = encode_sequence_numba(seq)  # Encode each sequence

    return encoded_array

def annotate_new_data_parallel(new_encoded_data, model):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Pad sequences (though they are already fixed-length)
    X_new_padded = pad_sequences(new_encoded_data, padding='post', dtype='int8')

    # Create a dataset for efficient input feeding
    dataset = tf.data.Dataset.from_tensor_slices(X_new_padded)
    dataset = dataset.batch(2048)  # Optimize batch size
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Run inference (TensorFlow will auto-distribute across GPUs)
    predictions = model.predict(dataset)
    return predictions

def annotate_new_data(new_encoded_data, model):
    X_new_padded = pad_sequences(new_encoded_data, padding='post', dtype='int8')
    predictions = model.predict(X_new_padded)
    return predictions

# Function to calculate the total number of rows in the Parquet file
def calculate_total_rows(parquet_file):
    df = pl.scan_parquet(parquet_file)
    total_rows = df.collect().shape[0]
    return total_rows

# Modified function to estimate the average read length from the bin name
def estimate_average_read_length_from_bin(bin_name):
    bounds = bin_name.replace("bp", "").split("_")
    lower_bound = int(bounds[0])
    upper_bound = int(bounds[1])
    return (lower_bound + upper_bound) / 2

# Function to calculate the total number of rows in the Parquet file
def calculate_total_rows(parquet_file):
    df = pl.scan_parquet(parquet_file)
    total_rows = df.collect().shape[0]
    return total_rows

# Modified function to estimate the average read length from the bin name
def estimate_average_read_length_from_bin(bin_name):
    bounds = bin_name.replace("bp", "").split("_")
    lower_bound = int(bounds[0])
    upper_bound = int(bounds[1])
    return (lower_bound + upper_bound) / 2

def model_predictions(parquet_file, chunk_start, chunk_size, model_path):
    
    total_rows = calculate_total_rows(parquet_file)
    bin_name = os.path.basename(parquet_file).replace(".parquet", "")
    
    # Estimate the average read length from the bin name and adjust chunk size
    estimated_avg_length = estimate_average_read_length_from_bin(bin_name)
    dynamic_chunk_size = int(chunk_size * (500 / estimated_avg_length))  # Scale chunk size dynamically

    scan_df = pl.scan_parquet(parquet_file)

    # Read the input file in chunks
    num_chunks = (total_rows // dynamic_chunk_size) + (1 if total_rows % dynamic_chunk_size > 0 else 0)

    strategy = tf.distribute.MirroredStrategy()  
    with strategy.scope():
        model = load_model(model_path)

    # Iterate over chunks within the Parquet file
    for chunk_idx in range(chunk_start, num_chunks + 1):
        print(f"Processing {bin_name}: chunk {chunk_idx}")
        
        # Read the current chunk of rows from the Parquet file
        df_chunk = scan_df.slice((chunk_idx - 1) * dynamic_chunk_size, dynamic_chunk_size).collect()
        read_names = df_chunk["ReadName"].to_list()

        reads = df_chunk["read"].to_list()
        read_lengths = df_chunk["read_length"].to_list()

        model = load_model(model_path) if len(reads) < 100 else model

        logger.info("Encoding and padding sequences")
        encoded_data = preprocess_sequences(reads)
        logger.info("Encoding and padding sequences finished")

        logger.info(f"Inferring labels")
        chunk_predictions = annotate_new_data_parallel(encoded_data, model) if len(reads) >= 100 else annotate_new_data(encoded_data, model)
        logger.info(f"labels inferred")

        del df_chunk, encoded_data
        gc.collect()

        yield parquet_file, chunk_idx, chunk_predictions, read_names, reads, read_lengths

