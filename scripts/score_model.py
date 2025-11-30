"""
Model scoring utilities for sequence labeling evaluation.

This module provides functions to calculate comprehensive metrics for evaluating
trained models on validation/test data, including:
- Per-segment precision, recall, F1-score
- Overall macro/micro-averaged metrics
- Confusion matrices
- Sequence-level accuracy
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def flatten_sequences(y_true_seqs, y_pred_seqs, label_names=None):
    """
    Flatten sequences of class indices, removing padding positions.

    Args:
        y_true_seqs: List of 1D arrays of true class indices
        y_pred_seqs: List of 1D arrays of predicted class indices
        label_names: Optional list of label names (for validation)

    Returns:
        y_true_flat: 1D array of true class indices
        y_pred_flat: 1D array of predicted class indices
        valid_lengths: List of valid (non-padding) length for each sequence
    """
    y_true_flat = []
    y_pred_flat = []
    valid_lengths = []

    for y_true, y_pred in zip(y_true_seqs, y_pred_seqs):
        # Ensure 1D arrays (class indices)
        if y_true.ndim != 1:
            raise ValueError(f"Expected 1D array of class indices, got shape {y_true.shape}")
        if y_pred.ndim != 1:
            raise ValueError(f"Expected 1D array of class indices, got shape {y_pred.shape}")

        # Find valid positions (non-padding)
        # Padding positions have class index 0
        valid_mask = y_true != 0

        # Only include valid (non-padding) positions
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        y_true_flat.extend(y_true_valid)
        y_pred_flat.extend(y_pred_valid)
        valid_lengths.append(len(y_true_valid))

    return np.array(y_true_flat), np.array(y_pred_flat), valid_lengths


def calculate_per_segment_metrics(y_true, y_pred, label_names):
    """
    Calculate precision, recall, and F1-score for each segment type.

    Args:
        y_true: 1D array of true class indices
        y_pred: 1D array of predicted class indices
        label_names: List of label names ordered by class index

    Returns:
        DataFrame with per-segment metrics
    """
    # Calculate precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Build results dataframe
    results = []
    for idx, label in enumerate(label_names):
        if idx < len(precision):  # Skip if no predictions for this class
            results.append({
                'segment': label,
                'precision': precision[idx],
                'recall': recall[idx],
                'f1_score': f1[idx],
                'support': support[idx]
            })

    return pd.DataFrame(results)


def calculate_overall_metrics(y_true, y_pred):
    """
    Calculate macro and micro-averaged metrics.

    Args:
        y_true: 1D array of true class indices
        y_pred: 1D array of predicted class indices

    Returns:
        Dictionary with overall metrics
    """
    # Micro-averaged (all positions weighted equally)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )

    # Macro-averaged (all segments weighted equally)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'accuracy': accuracy
    }


def calculate_sequence_level_accuracy(y_true_seqs, y_pred_seqs):
    """
    Calculate the percentage of sequences that are perfectly predicted.

    Args:
        y_true_seqs: List of 1D arrays of true class indices
        y_pred_seqs: List of 1D arrays of predicted class indices

    Returns:
        float: Percentage of perfectly predicted sequences
    """
    perfect_count = 0
    total_count = len(y_true_seqs)

    for y_true, y_pred in zip(y_true_seqs, y_pred_seqs):
        # Check if entire sequence matches (including padding positions)
        if np.array_equal(y_true, y_pred):
            perfect_count += 1

    return 100.0 * perfect_count / total_count if total_count > 0 else 0.0


def generate_confusion_matrix(y_true, y_pred, label_names):
    """
    Generate confusion matrix for segment predictions.

    Args:
        y_true: 1D array of true class indices
        y_pred: 1D array of predicted class indices
        label_names: List of label names

    Returns:
        DataFrame: Confusion matrix with labeled rows/columns
    """
    cm = confusion_matrix(y_true, y_pred)

    # Create labeled dataframe
    cm_df = pd.DataFrame(
        cm,
        index=[f'True_{label}' for label in label_names],
        columns=[f'Pred_{label}' for label in label_names]
    )

    return cm_df


def plot_confusion_matrix(cm_df, output_path=None, figsize=(12, 10)):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm_df: DataFrame confusion matrix
        output_path: Optional path to save figure
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize by row (true labels) to show percentages
    cm_normalized = cm_df.div(cm_df.sum(axis=1), axis=0) * 100

    # Plot heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                cbar_kws={'label': 'Percentage (%)'}, ax=ax)

    ax.set_title('Confusion Matrix (Row-Normalized)', fontsize=14, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_per_segment_metrics(metrics_df, output_path=None):
    """
    Create bar plots for per-segment precision, recall, and F1-score.

    Args:
        metrics_df: DataFrame with per-segment metrics
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ['precision', 'recall', 'f1_score']
    titles = ['Precision', 'Recall', 'F1-Score']

    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.bar(metrics_df['segment'], metrics_df[metric],
                     color='skyblue', edgecolor='black')

        # Color bars based on score (red < 0.7, yellow 0.7-0.9, green > 0.9)
        for bar, score in zip(bars, metrics_df[metric]):
            if score < 0.7:
                bar.set_color('lightcoral')
            elif score < 0.9:
                bar.set_color('khaki')
            else:
                bar.set_color('lightgreen')

        ax.set_ylabel(title, fontsize=12)
        ax.set_xlabel('Segment', fontsize=12)
        ax.set_title(f'{title} by Segment', fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def generate_classification_report(y_true, y_pred, label_names, output_path=None):
    """
    Generate a detailed classification report.

    Args:
        y_true: 1D array of true class indices
        y_pred: 1D array of predicted class indices
        label_names: List of label names
        output_path: Optional path to save report as text file

    Returns:
        str: Classification report
    """
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        digits=4,
        zero_division=0
    )

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


def create_comprehensive_evaluation_report(y_true_seqs, y_pred_seqs, label_names,
                                          output_dir, model_name):
    """
    Generate a comprehensive evaluation report with all metrics and visualizations.

    Args:
        y_true_seqs: List of arrays of true labels
        y_pred_seqs: List of arrays of predicted labels
        label_names: List of label names
        output_dir: Directory to save outputs
        model_name: Name of the model being evaluated

    Returns:
        Dictionary containing all computed metrics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Flatten sequences
    y_true_flat, y_pred_flat, valid_lengths = flatten_sequences(
        y_true_seqs, y_pred_seqs, label_names
    )

    print(f"Evaluating {len(y_true_seqs)} sequences...")
    print(f"Total valid positions: {len(y_true_flat):,}")
    print(f"Average sequence length: {np.mean(valid_lengths):.1f}")

    # Calculate all metrics
    per_segment_metrics = calculate_per_segment_metrics(
        y_true_flat, y_pred_flat, label_names
    )
    overall_metrics = calculate_overall_metrics(y_true_flat, y_pred_flat)
    seq_accuracy = calculate_sequence_level_accuracy(y_true_seqs, y_pred_seqs)
    cm_df = generate_confusion_matrix(y_true_flat, y_pred_flat, label_names)

    # Save metrics to files
    per_segment_metrics.to_csv(
        f'{output_dir}/{model_name}_per_segment_metrics.tsv',
        sep='\t', index=False, float_format='%.4f'
    )

    overall_df = pd.DataFrame([overall_metrics])
    overall_df['sequence_accuracy'] = seq_accuracy
    overall_df.to_csv(
        f'{output_dir}/{model_name}_overall_metrics.tsv',
        sep='\t', index=False, float_format='%.4f'
    )

    cm_df.to_csv(
        f'{output_dir}/{model_name}_confusion_matrix.tsv',
        sep='\t', float_format='%d'
    )

    # Generate classification report
    report = generate_classification_report(
        y_true_flat, y_pred_flat, label_names,
        output_path=f'{output_dir}/{model_name}_classification_report.txt'
    )

    # Create visualizations
    with PdfPages(f'{output_dir}/{model_name}_evaluation_plots.pdf') as pdf:
        # Per-segment metrics plot
        fig1 = plot_per_segment_metrics(per_segment_metrics)
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)

        # Confusion matrix plot
        fig2 = plot_confusion_matrix(cm_df)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)

        # Overall metrics summary plot
        fig3, ax = plt.subplots(figsize=(10, 6))
        metrics_to_plot = {
            'Micro F1': overall_metrics['micro_f1'],
            'Macro F1': overall_metrics['macro_f1'],
            'Micro Precision': overall_metrics['micro_precision'],
            'Macro Precision': overall_metrics['macro_precision'],
            'Micro Recall': overall_metrics['micro_recall'],
            'Macro Recall': overall_metrics['macro_recall'],
            'Accuracy': overall_metrics['accuracy'],
            'Seq Accuracy (%)': seq_accuracy / 100
        }

        bars = ax.barh(list(metrics_to_plot.keys()), list(metrics_to_plot.values()),
                      color='steelblue', edgecolor='black')
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title(f'Overall Model Performance: {model_name}', fontsize=14)
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.4f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)

    # Print summary
    print("\n" + "="*80)
    print(f"EVALUATION SUMMARY: {model_name}")
    print("="*80)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {overall_metrics['accuracy']:.4f}")
    print(f"  Micro F1:           {overall_metrics['micro_f1']:.4f}")
    print(f"  Macro F1:           {overall_metrics['macro_f1']:.4f}")
    print(f"  Sequence Accuracy:  {seq_accuracy:.2f}%")
    print(f"\nPer-Segment F1 Scores:")
    for _, row in per_segment_metrics.iterrows():
        print(f"  {row['segment']:12s}  {row['f1_score']:.4f}  (n={int(row['support']):,})")
    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}/")
    print("="*80)

    return {
        'per_segment_metrics': per_segment_metrics,
        'overall_metrics': overall_metrics,
        'sequence_accuracy': seq_accuracy,
        'confusion_matrix': cm_df,
        'classification_report': report
    }
