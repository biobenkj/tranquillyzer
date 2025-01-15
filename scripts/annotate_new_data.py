import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import multiprocessing as mp
import os

# Enable memory growth to avoid pre-allocating all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join(complement[base] for base in reversed(seq))

nucleotide_to_id = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}

def encode_sequence(read, nucleotide_to_id):
    return [nucleotide_to_id[n] for n in read]

def annotate_new_data(new_data, model_path):
    # Encode the sequence and predict the labels

    model = load_model(model_path)

    MAX_READ_LENGTH = max([len(seq) for seq in new_data])

    X_new_encoded = [encode_sequence(read, nucleotide_to_id) for read in new_data]
    X_new_padded = pad_sequences(X_new_encoded, padding='post', dtype='int8')

    predictions = model.predict(X_new_padded)
    return predictions

def annotate_new_data_parallel(new_data, model_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Set up the mirrored strategy to use all available GPUs
    strategy = tf.distribute.MirroredStrategy()

    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Load the model within the scope of the strategy
    with strategy.scope():
        model = load_model(model_path)

        # Prepare the input data (no need to split manually, TensorFlow will handle it)
        MAX_READ_LENGTH = max([len(seq) for seq in new_data])

        X_new_encoded = [encode_sequence(read, nucleotide_to_id) for read in new_data]
        X_new_padded = pad_sequences(X_new_encoded, padding='post')

        # Create a dataset and leverage TensorFlow's input pipeline
        dataset = tf.data.Dataset.from_tensor_slices(X_new_padded)
        dataset = dataset.batch(2048)  # Adjust batch size as necessary
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Run inference (TensorFlow will automatically split this across GPUs)
        predictions = model.predict(dataset)

    return predictions