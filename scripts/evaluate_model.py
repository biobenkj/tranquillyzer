#!/usr/bin/env python3
"""
Evaluate a trained tranquillyzer model on validation/test data.

This script loads a trained model and evaluates it on validation data,
generating comprehensive metrics including:
- Per-segment precision, recall, F1-score
- Overall macro/micro-averaged metrics
- Confusion matrix
- Sequence-level accuracy
- Visualization plots

Usage:
    python evaluate_model.py --model_dir MODEL_DIR [--validation_data VAL_DATA]
                             [--output_dir OUTPUT_DIR]

    python evaluate_model.py --model_dir output/my_model_0
                             --validation_data output/simulated_data/validation_reads.pkl
                             --output_dir evaluation_results/
"""

import os
import sys
import argparse
import pickle
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add parent directory to path for imports
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from scripts.score_model import create_comprehensive_evaluation_report
from scripts.train_new_model import (
    ont_read_annotator,
    DynamicPaddingDataGenerator
)
from scripts.annotate_new_data import (
    annotate_new_data_parallel,
    preprocess_sequences
)
from scripts.extract_annotated_seqs import extract_annotated_full_length_seqs
from sklearn.preprocessing import LabelBinarizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_metadata(model_dir):
    """
    Load trained model, label binarizer, and training parameters.

    Args:
        model_dir: Directory containing the trained model files

    Returns:
        Tuple of (model, label_binarizer, params, model_name, has_crf)
    """
    model_dir = Path(model_dir)

    # Find model files
    h5_files = list(model_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 model file found in {model_dir}")

    model_file = h5_files[0]
    model_name = model_file.stem

    # Check if CRF model
    has_crf = "_w_CRF" in model_name

    # Load label binarizer
    lb_file = model_dir / f"{model_name}_lbl_bin.pkl"
    if not lb_file.exists():
        # Try without _w_CRF suffix
        base_name = model_name.replace("_w_CRF", "")
        lb_file = model_dir / f"{base_name}_lbl_bin.pkl"

    if not lb_file.exists():
        raise FileNotFoundError(f"Label binarizer not found: {lb_file}")

    with open(lb_file, 'rb') as f:
        label_binarizer = pickle.load(f)

    # Load parameters
    param_file = model_dir / f"{model_name}_params.json"
    if not param_file.exists():
        base_name = model_name.replace("_w_CRF", "")
        param_file = model_dir / f"{base_name}_params.json"

    if not param_file.exists():
        logger.warning(f"Parameters file not found: {param_file}")
        params = None
    else:
        import json
        with open(param_file, 'r') as f:
            params = json.load(f)

    # Build and load model
    if params:
        num_labels = len(label_binarizer.classes_)
        model = ont_read_annotator(
            vocab_size=int(params.get("vocab_size", 6)),
            embedding_dim=int(params.get("embedding_dim", 64)),
            num_labels=num_labels,
            conv_layers=int(params.get("conv_layers", 3)),
            conv_filters=int(params.get("conv_filters", 260)),
            conv_kernel_size=int(params.get("conv_kernel_size", 25)),
            lstm_layers=int(params.get("lstm_layers", 1)),
            lstm_units=int(params.get("lstm_units", 128)),
            bidirectional=params.get("bidirectional", "true").lower() == "true",
            crf_layer=has_crf,
            attention_heads=int(params.get("attention_heads", 0)),
            dropout_rate=float(params.get("dropout_rate", 0.35)),
            regularization=float(params.get("regularization", 0.01)),
            learning_rate=float(params.get("learning_rate", 0.01))
        )

        if has_crf:
            # For CRF models, need to build the model first
            dummy_input = tf.zeros((1, 512), dtype=tf.int32)
            _ = model(dummy_input)
            model.load_weights(str(model_file))
        else:
            model = tf.keras.models.load_model(str(model_file))
    else:
        # Try to load model directly
        model = tf.keras.models.load_model(str(model_file))

    logger.info(f"Loaded model: {model_name}")
    logger.info(f"CRF model: {has_crf}")
    logger.info(f"Number of labels: {len(label_binarizer.classes_)}")
    logger.info(f"Labels: {label_binarizer.classes_}")

    return model, label_binarizer, params, model_name, has_crf


def load_validation_data(validation_data_path):
    """
    Load validation reads and labels.

    Args:
        validation_data_path: Path to validation data pickle file or directory

    Returns:
        Tuple of (reads, labels)
    """
    validation_path = Path(validation_data_path)

    # If directory provided, look for standard validation files
    if validation_path.is_dir():
        reads_file = validation_path / "validation_reads.pkl"
        labels_file = validation_path / "validation_labels.pkl"

        if not reads_file.exists():
            reads_file = validation_path / "reads.pkl"
        if not labels_file.exists():
            labels_file = validation_path / "labels.pkl"

        if not reads_file.exists():
            raise FileNotFoundError(f"Validation reads not found in {validation_path}")
        if not labels_file.exists():
            raise FileNotFoundError(f"Validation labels not found in {validation_path}")

        with open(reads_file, 'rb') as f:
            reads = pickle.load(f)
        with open(labels_file, 'rb') as f:
            labels = pickle.load(f)

    # If file provided, assume it contains both reads and labels
    elif validation_path.suffix == '.pkl':
        with open(validation_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            reads = data.get('reads') or data.get('validation_reads')
            labels = data.get('labels') or data.get('validation_labels')
        elif isinstance(data, tuple) and len(data) == 2:
            reads, labels = data
        else:
            raise ValueError(f"Unexpected validation data format in {validation_path}")
    else:
        raise ValueError(f"Invalid validation data path: {validation_path}")

    logger.info(f"Loaded {len(reads)} validation sequences")
    logger.info(f"Average sequence length: {np.mean([len(r) for r in reads]):.1f}")

    return reads, labels


def generate_predictions(model, reads, label_binarizer, has_crf,
                        max_batch_size=1024, min_batch_size=16):
    """
    Generate model predictions on validation reads.

    Args:
        model: Trained TensorFlow model
        reads: List of read sequences
        label_binarizer: Fitted LabelBinarizer
        has_crf: Whether model has CRF layer
        max_batch_size: Maximum batch size for inference
        min_batch_size: Minimum batch size for inference

    Returns:
        predictions: List of predicted label sequences
    """
    logger.info("Generating predictions...")

    # Preprocess sequences
    encoded_data = preprocess_sequences(reads)

    # Generate predictions
    predictions = annotate_new_data_parallel(
        encoded_data, model,
        max_batch_size,
        min_batch=min_batch_size,
        strategy=None
    )

    logger.info(f"Generated {len(predictions)} predictions")

    return predictions


def convert_labels_to_indices(labels, label_binarizer):
    """
    Convert list of label sequences (strings) to class indices.

    Args:
        labels: List of label sequences (lists of strings)
        label_binarizer: Fitted LabelBinarizer

    Returns:
        List of numpy arrays containing class indices
    """
    # Create mapping from label string to index
    label_to_idx = {label: idx for idx, label in enumerate(label_binarizer.classes_)}

    y_true_sequences = []
    for label_seq in labels:
        # Convert string labels to indices
        indices = np.array([label_to_idx[label] for label in label_seq])
        y_true_sequences.append(indices)

    return y_true_sequences


def convert_predictions_to_indices(predictions):
    """
    Convert model predictions to class indices.

    Args:
        predictions: List of prediction arrays (can be class indices, logits, or probabilities)

    Returns:
        List of numpy arrays containing class indices
    """
    y_pred_sequences = []

    for pred in predictions:
        if pred.ndim == 1:
            # Already class indices
            y_pred_sequences.append(pred)
        elif pred.ndim == 2:
            # Probabilities or logits - take argmax
            pred_indices = np.argmax(pred, axis=1)
            y_pred_sequences.append(pred_indices)
        else:
            raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    return y_pred_sequences


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained tranquillyzer model on validation data'
    )
    parser.add_argument('--model_dir', '-m', required=True,
                       help='Directory containing the trained model')
    parser.add_argument('--validation_data', '-v',
                       help='Path to validation data (pickle file or directory). '
                            'If not provided, will look for validation data in model_dir parent.')
    parser.add_argument('--output_dir', '-o',
                       help='Directory to save evaluation results. '
                            'If not provided, creates "evaluation" subdirectory in model_dir.')
    parser.add_argument('--max_batch_size', type=int, default=1024,
                       help='Maximum batch size for inference (default: 1024)')
    parser.add_argument('--min_batch_size', type=int, default=16,
                       help='Minimum batch size for inference (default: 16)')

    args = parser.parse_args()

    # Load model and metadata
    model, label_binarizer, params, model_name, has_crf = load_model_and_metadata(
        args.model_dir
    )

    # Determine validation data path
    if args.validation_data:
        validation_data_path = args.validation_data
    else:
        # Look for validation data in parent directory of model_dir
        parent_dir = Path(args.model_dir).parent
        simulated_data_dir = parent_dir / "simulated_data"

        if simulated_data_dir.exists():
            validation_data_path = simulated_data_dir
        else:
            raise ValueError(
                "No validation data provided and could not find simulated_data directory. "
                "Please specify --validation_data"
            )

    # Load validation data
    reads, labels = load_validation_data(validation_data_path)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.model_dir) / "evaluation"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate predictions
    predictions = generate_predictions(
        model, reads, label_binarizer, has_crf,
        args.max_batch_size, args.min_batch_size
    )

    # Convert true labels to class indices
    y_true_sequences = convert_labels_to_indices(labels, label_binarizer)

    # Convert predictions to class indices
    y_pred_sequences = convert_predictions_to_indices(predictions)

    # Generate comprehensive evaluation report
    logger.info("Computing metrics...")
    results = create_comprehensive_evaluation_report(
        y_true_sequences,
        y_pred_sequences,
        label_binarizer.classes_,
        str(output_dir),
        model_name
    )

    logger.info(f"\nEvaluation complete! Results saved to: {output_dir}")

    return results


if __name__ == '__main__':
    main()
