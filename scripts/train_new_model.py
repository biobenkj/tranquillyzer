from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, BatchNormalization, Dropout, 
    Bidirectional, LSTM, Dense, TimeDistributed, Add, MultiHeadAttention
)
import pickle
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Mapping nucleotides to integers
nucleotide_to_id = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}

def encode_sequence(sequence):
    """Convert nucleotide sequence to list of integers."""
    return [nucleotide_to_id[base] for base in sequence]

class DynamicPaddingDataGenerator(Sequence):
    def __init__(self, X, Y, batch_size, label_binarizer):
        self.X = [encode_sequence(seq) for seq in X]
        self.Y = [label_binarizer.transform(labels) for labels in Y]
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_Y = self.Y[idx * self.batch_size:(idx + 1) * self.batch_size]
        max_len = max(len(x) for x in batch_X)
        X_padded = pad_sequences(batch_X, maxlen=max_len, padding="post", value=0)
        Y_padded = pad_sequences(batch_Y, maxlen=max_len, padding="post", value=0)
        return X_padded, Y_padded

def ont_read_annotator(
    vocab_size, 
    embedding_dim, 
    num_labels, 
    conv_layers=3, 
    conv_filters=260, 
    conv_kernel_size=25, 
    lstm_layers=1, 
    lstm_units=128, 
    bidirectional=True, 
    attention_heads=0, 
    dropout_rate=0.35, 
    regularization=0.01
):
    """
    Generalized ONT Read Annotator Model with configurable CNN, LSTM, and Attention layers.

    Parameters:
        vocab_size (int): Size of the input vocabulary.
        embedding_dim (int): Dimensionality of the embedding layer.
        num_labels (int): Number of output labels for classification.
        conv_layers (int): Number of Conv1D layers.
        conv_filters (int): Number of filters in each Conv1D layer.
        conv_kernel_size (int): Kernel size for Conv1D layers.
        lstm_layers (int): Number of LSTM layers.
        lstm_units (int): Number of units in LSTM layers.
        bidirectional (bool): Whether to use Bidirectional LSTM.
        attention_heads (int): Number of multi-head attention heads (0 to disable).
        dropout_rate (float): Dropout rate for regularization.
        regularization (float): L2 regularization strength.

    Returns:
        Model: Compiled TensorFlow Keras model.
    """

    inputs = Input(shape=(None,))
    x = Embedding(vocab_size, embedding_dim)(inputs)

    # Multi-scale convolutional layers
    for _ in range(conv_layers):
        x = Conv1D(
            filters=conv_filters, 
            kernel_size=conv_kernel_size, 
            activation="relu",
            padding="same", 
            kernel_regularizer=l2(regularization)
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    # LSTM layers
    for i in range(lstm_layers):
        lstm_layer = LSTM(
            lstm_units if i == 0 else lstm_units // 2,  # Reduce units in deeper layers
            return_sequences=True,
            kernel_regularizer=l2(regularization),
            recurrent_regularizer=l2(regularization)
        )
        x = Bidirectional(lstm_layer)(x) if bidirectional else lstm_layer(x)
        x = Dropout(dropout_rate)(x)

    # Optional Multi-Head Attention
    if attention_heads > 0:
        attention_output = MultiHeadAttention(num_heads=attention_heads, key_dim=lstm_units)(x, x)
        x = Add()([x, attention_output])  # Residual connection
        x = Dropout(dropout_rate)(x)

    # TimeDistributed output layer
    outputs = TimeDistributed(Dense(num_labels, activation="softmax", kernel_regularizer=l2(regularization)))(x)

    return Model(inputs, outputs)

