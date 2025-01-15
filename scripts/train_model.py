import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, Conv1D, LSTM, Bidirectional,
                                     TimeDistributed, Dense, Dropout, BatchNormalization,
                                     Add, MultiHeadAttention, MaxPooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import regularizers
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.utils import shuffle
from collections import Counter
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

class ReadAnnotator:
    def __init__(self, input_length, embedding_dim, num_labels, 
                 reg_param=0.001, num_conv_layers=4, conv_filters=270, 
                 conv_kernel_size=17, num_lstm_layers=1, lstm_units=128):
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.reg_param = reg_param
        self.num_conv_layers = num_conv_layers
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.num_lstm_layers = num_lstm_layers
        self.lstm_units = lstm_units
        self.model = self.build_model()

    def conv_block(self, inputs, filters, kernel_size):
        """Creates a convolutional block with Conv1D, BatchNormalization, and Dropout layers."""
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        return x

    def lstm_block(self, inputs):
        """Creates a series of Bidirectional LSTM layers."""
        x = inputs
        for _ in range(self.num_lstm_layers):
            x = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout=0.2))(x)
        return x

    def build_model(self):
        """Builds and returns the ONT read annotator model."""
        # Input layer
        sequence_input = Input(shape=(None,))
        
        # Embedding layer
        embedded_sequence = Embedding(6,  # +1 for 0 padding
                                      self.embedding_dim,
                                      input_length=self.input_length)(sequence_input)

        # Convolutional layers
        x = embedded_sequence
        for _ in range(self.num_conv_layers):
            x = self.conv_block(x, filters=self.conv_filters, kernel_size=self.conv_kernel_size)

        # LSTM layers
        x = self.lstm_block(x)

        # TimeDistributed Dense for output
        predictions = TimeDistributed(Dense(self.num_labels, activation='softmax'))(x)

        # Model definition
        model = Model(inputs=sequence_input, outputs=predictions)
        return model

    def compile_model(self, learning_rate=0.001, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        """Compiles the model with the given optimizer, loss, metrics, and learning rate."""
        if optimizer == 'adam':
            optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer_instance = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError("Unsupported optimizer type. Use 'adam', 'sgd', or 'rmsprop'.")

        self.model.compile(optimizer=optimizer_instance, loss=loss, metrics=metrics)

    def summary(self):
        """Prints a summary of the model."""
        self.model.summary()

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_X = self.X[start:end]
        batch_Y = self.Y[start:end]
        return batch_X, batch_Y

def compute_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}

def train_model(X_train, Y_train, X_val, Y_val, model, epochs=5, batch_size=150):
    X_train, Y_train = shuffle(X_train, Y_train)
    X_val, Y_val = shuffle(X_val, Y_val)

    # Compute class weights
    y_train_flat = np.argmax(Y_train, axis=-1).flatten()
    class_weight_dict = compute_weights(y_train_flat)

    # Define callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                                  factor=0.1, 
                                  patience=0, 
                                  verbose=1, 
                                  min_lr=1e-8)

    early_stop = EarlyStopping(monitor='val_accuracy', 
                               min_delta=0, 
                               patience=5, 
                               verbose=1, 
                               mode='max', 
                               baseline=0.995,
                               restore_best_weights=True)

    # Prepare data generators
    train_generator = DataGenerator(X_train, Y_train, batch_size)
    val_generator = DataGenerator(X_val, Y_val, batch_size)

    # Use fit for training
    history = model.fit(x=train_generator, 
                        epochs=epochs, 
                        validation_data=val_generator,
                        class_weight=class_weight_dict,
                        callbacks=[reduce_lr, early_stop])

    return model, history

EMBEDDING_DIM = 6
NUM_LABELS = 6