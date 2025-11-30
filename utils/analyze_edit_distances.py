#!/usr/bin/env python3
"""
Analyze edit distance distributions for valid reads with ONT-specific thresholds.

This script reads the annotations parquet file and generates:
1. Summary statistics for edit distances per segment
2. ONT-aware filtering thresholds based on segment length and expected error rates
3. Visualizations of edit distance distributions
4. Recommendations for filtering thresholds

Usage:
    python analyze_edit_distances.py <parquet_file> [--output_dir OUTPUT_DIR] [--ont_error_rate ONT_ERROR_RATE]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from math import ceil

sns.set_style("whitegrid")

def load_valid_reads(parquet_file):
    """Load valid reads from parquet file."""
    print(f"Loading data from {parquet_file}...")

    # Try polars first (faster), fall back to pandas if needed
    try:
        import polars as pl
        df = pl.read_parquet(parquet_file)
        # Convert to pandas for easier manipulation
        df = df.to_pandas()
    except Exception as e:
        print(f"Polars not available or failed ({e}), using pandas...")
        df = pd.read_parquet(parquet_file)

    print(f"Loaded {len(df):,} reads")
    return df

def identify_edit_distance_columns(df):
    """Identify all edit distance columns in the DataFrame."""
    edit_dist_cols = [col for col in df.columns if col.endswith('_edit_distance')]

    # Categorize by segment type
    barcode_segments = []
    fixed_segments = []

    for col in edit_dist_cols:
        segment = col.replace('_edit_distance', '')
        if segment in ['CBC', 'i7', 'i5']:
            barcode_segments.append(segment)
        else:
            fixed_segments.append(segment)

    return edit_dist_cols, barcode_segments, fixed_segments

def get_segment_lengths(df, segments):
    """Calculate median segment length for each segment."""
    segment_lengths = {}

    for segment in segments:
        start_col = f'{segment}_Starts'
        end_col = f'{segment}_Ends'

        if start_col in df.columns and end_col in df.columns:
            # Calculate lengths where both start and end are present
            lengths = df[end_col] - df[start_col]
            lengths = lengths[lengths > 0]  # Filter out invalid lengths

            if len(lengths) > 0:
                segment_lengths[segment] = lengths.median()
            else:
                # Fallback: estimate from sequence column if available
                seq_col = f'{segment}_Sequences'
                if seq_col in df.columns:
                    seq_lengths = df[seq_col].dropna().str.len()
                    if len(seq_lengths) > 0:
                        segment_lengths[segment] = seq_lengths.median()

    return segment_lengths

def calculate_ont_threshold(segment_length, error_rate):
    """
    Calculate acceptable edit distance based on ONT error rate and segment length.

    For ONT data with error_rate (e.g., 0.05 for 5%), the maximum acceptable
    edit distance is: ceil(segment_length * error_rate)
    """
    if segment_length is None or segment_length == 0:
        return None
    return ceil(segment_length * error_rate)

def calculate_statistics(df, segment, edit_dist_col, segment_length=None, ont_error_rate=0.05):
    """Calculate comprehensive statistics for a segment's edit distances."""
    values = df[edit_dist_col].dropna()

    if len(values) == 0:
        return None

    # Calculate ONT-aware threshold if segment length is available
    ont_threshold_strict = calculate_ont_threshold(segment_length, 0.02) if segment_length else None  # 2% error
    ont_threshold_permissive = calculate_ont_threshold(segment_length, ont_error_rate) if segment_length else None  # User-specified (default 5%)
    ont_threshold_very_permissive = calculate_ont_threshold(segment_length, 0.10) if segment_length else None  # 10% error

    stats = {
        'segment': segment,
        'median_length': segment_length if segment_length else np.nan,
        'total_reads': len(df),
        'reads_with_segment': len(values),
        'reads_missing_segment': len(df) - len(values),
        'percent_detected': 100 * len(values) / len(df),

        # Basic statistics
        'mean': values.mean(),
        'median': values.median(),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),

        # Percentiles
        'p25': values.quantile(0.25),
        'p75': values.quantile(0.75),
        'p95': values.quantile(0.95),
        'p99': values.quantile(0.99),

        # Quality metrics
        'perfect_matches': (values == 0).sum(),
        'percent_perfect': 100 * (values == 0).sum() / len(values),
        'edit_dist_1_or_less': (values <= 1).sum(),
        'percent_ed1_or_less': 100 * (values <= 1).sum() / len(values),
        'edit_dist_3_or_less': (values <= 3).sum(),
        'percent_ed3_or_less': 100 * (values <= 3).sum() / len(values),
        'edit_dist_5_or_less': (values <= 5).sum(),
        'percent_ed5_or_less': 100 * (values <= 5).sum() / len(values),

        # ONT-specific thresholds
        'ont_threshold_2pct': ont_threshold_strict,
        'ont_threshold_5pct': ont_threshold_permissive,
        'ont_threshold_10pct': ont_threshold_very_permissive,
    }

    # Calculate percentage of reads passing ONT thresholds
    if ont_threshold_strict is not None:
        stats['percent_passing_2pct'] = 100 * (values <= ont_threshold_strict).sum() / len(values)
    else:
        stats['percent_passing_2pct'] = np.nan

    if ont_threshold_permissive is not None:
        stats['percent_passing_5pct'] = 100 * (values <= ont_threshold_permissive).sum() / len(values)
    else:
        stats['percent_passing_5pct'] = np.nan

    if ont_threshold_very_permissive is not None:
        stats['percent_passing_10pct'] = 100 * (values <= ont_threshold_very_permissive).sum() / len(values)
    else:
        stats['percent_passing_10pct'] = np.nan

    return stats

def plot_edit_distance_distributions(df, segments, segment_lengths, ont_error_rate, output_pdf):
    """Create comprehensive visualizations of edit distance distributions."""
    print(f"Creating visualizations in {output_pdf}...")

    with PdfPages(output_pdf) as pdf:
        # 1. Individual histograms for each segment
        for segment in segments:
            edit_dist_col = f'{segment}_edit_distance'
            if edit_dist_col not in df.columns:
                continue

            values = df[edit_dist_col].dropna()
            if len(values) == 0:
                continue

            segment_length = segment_lengths.get(segment)
            ont_threshold_2pct = calculate_ont_threshold(segment_length, 0.02)
            ont_threshold_5pct = calculate_ont_threshold(segment_length, ont_error_rate)
            ont_threshold_10pct = calculate_ont_threshold(segment_length, 0.10)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Histogram
            ax1.hist(values, bins=min(50, int(values.max()) + 1), edgecolor='black', alpha=0.7)
            ax1.axvline(values.median(), color='red', linestyle='--',
                       label=f'Median: {values.median():.1f}')

            if ont_threshold_2pct is not None:
                ax1.axvline(ont_threshold_2pct, color='green', linestyle='--',
                           linewidth=2, label=f'ONT 2% error: {ont_threshold_2pct}')
            if ont_threshold_5pct is not None:
                ax1.axvline(ont_threshold_5pct, color='orange', linestyle='--',
                           linewidth=2, label=f'ONT {int(ont_error_rate*100)}% error: {ont_threshold_5pct}')
            if ont_threshold_10pct is not None:
                ax1.axvline(ont_threshold_10pct, color='purple', linestyle='--',
                           linewidth=2, label=f'ONT 10% error: {ont_threshold_10pct}')

            ax1.set_xlabel('Edit Distance')
            ax1.set_ylabel('Frequency')
            title = f'{segment} Edit Distance Distribution'
            if segment_length:
                title += f'\n(median length: {segment_length:.0f} bp)'
            ax1.set_title(title)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Cumulative distribution
            sorted_vals = np.sort(values)
            cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax2.plot(sorted_vals, cumulative, linewidth=2)

            if ont_threshold_2pct is not None:
                pct_passing = 100 * (values <= ont_threshold_2pct).sum() / len(values)
                ax2.axvline(ont_threshold_2pct, color='green', linestyle='--',
                           linewidth=2, alpha=0.7,
                           label=f'ONT 2% threshold: {ont_threshold_2pct} ({pct_passing:.1f}% pass)')
            if ont_threshold_5pct is not None:
                pct_passing = 100 * (values <= ont_threshold_5pct).sum() / len(values)
                ax2.axvline(ont_threshold_5pct, color='orange', linestyle='--',
                           linewidth=2, alpha=0.7,
                           label=f'ONT {int(ont_error_rate*100)}% threshold: {ont_threshold_5pct} ({pct_passing:.1f}% pass)')
            if ont_threshold_10pct is not None:
                pct_passing = 100 * (values <= ont_threshold_10pct).sum() / len(values)
                ax2.axvline(ont_threshold_10pct, color='purple', linestyle='--',
                           linewidth=2, alpha=0.7,
                           label=f'ONT 10% threshold: {ont_threshold_10pct} ({pct_passing:.1f}% pass)')

            ax2.set_xlabel('Edit Distance')
            ax2.set_ylabel('Cumulative Fraction')
            ax2.set_title(f'{segment} Cumulative Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # 2. Comparison box plot (all segments)
        fig, ax = plt.subplots(figsize=(14, 6))

        plot_data = []
        plot_labels = []
        ont_thresholds_2pct = []
        ont_thresholds_5pct = []
        ont_thresholds_10pct = []

        for segment in segments:
            edit_dist_col = f'{segment}_edit_distance'
            if edit_dist_col in df.columns:
                values = df[edit_dist_col].dropna()
                if len(values) > 0:
                    plot_data.append(values)
                    plot_labels.append(segment)

                    segment_length = segment_lengths.get(segment)
                    ont_thresholds_2pct.append(calculate_ont_threshold(segment_length, 0.02))
                    ont_thresholds_5pct.append(calculate_ont_threshold(segment_length, ont_error_rate))
                    ont_thresholds_10pct.append(calculate_ont_threshold(segment_length, 0.10))

        if plot_data:
            bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                           showfliers=False)  # Hide outliers for clarity

            # Color barcode segments differently from fixed segments
            for i, (patch, label) in enumerate(zip(bp['boxes'], plot_labels)):
                if label in ['CBC', 'i7', 'i5']:
                    patch.set_facecolor('lightblue')
                else:
                    patch.set_facecolor('lightcoral')

            # Plot ONT thresholds as horizontal lines per segment
            x_positions = range(1, len(plot_labels) + 1)
            for i, (x, thr_2, thr_5, thr_10) in enumerate(zip(x_positions, ont_thresholds_2pct, ont_thresholds_5pct, ont_thresholds_10pct)):
                if thr_2 is not None:
                    ax.plot([x-0.3, x+0.3], [thr_2, thr_2], 'g-', linewidth=2, alpha=0.7)
                if thr_5 is not None:
                    ax.plot([x-0.3, x+0.3], [thr_5, thr_5], 'orange', linewidth=2, alpha=0.7)
                if thr_10 is not None:
                    ax.plot([x-0.3, x+0.3], [thr_10, thr_10], 'purple', linewidth=2, alpha=0.7)

            ax.set_ylabel('Edit Distance')
            ax.set_title('Edit Distance Comparison Across Segments')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

            # Add legend
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            legend_elements = [
                Patch(facecolor='lightblue', label='Barcode segments'),
                Patch(facecolor='lightcoral', label='Fixed segments'),
                Line2D([0], [0], color='green', linewidth=2, label='ONT 2% error threshold'),
                Line2D([0], [0], color='orange', linewidth=2, label=f'ONT {int(ont_error_rate*100)}% error threshold'),
                Line2D([0], [0], color='purple', linewidth=2, label='ONT 10% error threshold')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # 3. Error rate distribution plot (edit distance / segment length)
        fig, ax = plt.subplots(figsize=(12, 6))

        error_rate_data = []
        error_rate_labels = []

        for segment in segments:
            edit_dist_col = f'{segment}_edit_distance'
            segment_length = segment_lengths.get(segment)

            if edit_dist_col in df.columns and segment_length and segment_length > 0:
                values = df[edit_dist_col].dropna()
                if len(values) > 0:
                    # Calculate error rates as percentages
                    error_rates = (values / segment_length) * 100
                    error_rate_data.append(error_rates)
                    error_rate_labels.append(f'{segment}\n({segment_length:.0f}bp)')

        if error_rate_data:
            bp = ax.boxplot(error_rate_data, labels=error_rate_labels, patch_artist=True,
                           showfliers=False)

            # Color barcode segments differently
            for i, (patch, label) in enumerate(zip(bp['boxes'], error_rate_labels)):
                segment_name = label.split('\n')[0]
                if segment_name in ['CBC', 'i7', 'i5']:
                    patch.set_facecolor('lightblue')
                else:
                    patch.set_facecolor('lightcoral')

            # Add reference lines for 2%, 5%, and 10% error
            ax.axhline(2, color='green', linestyle='--', linewidth=2, alpha=0.7, label='2% error rate')
            ax.axhline(ont_error_rate * 100, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'{int(ont_error_rate*100)}% error rate')
            ax.axhline(10, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='10% error rate')

            ax.set_ylabel('Error Rate (%)')
            ax.set_title('Error Rate Distribution Across Segments\n(Edit Distance / Segment Length)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()

            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # 4. Heatmap of correlation between segment edit distances
        if len(segments) > 1:
            edit_dist_df = df[[f'{seg}_edit_distance' for seg in segments
                              if f'{seg}_edit_distance' in df.columns]].copy()
            edit_dist_df.columns = [col.replace('_edit_distance', '') for col in edit_dist_df.columns]

            if len(edit_dist_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = edit_dist_df.corr()
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                           square=True, ax=ax, cbar_kws={'label': 'Correlation'})
                ax.set_title('Edit Distance Correlation Between Segments')
                plt.tight_layout()
                pdf.savefig()
                plt.close()

def generate_filtering_recommendations(stats_df, ont_error_rate):
    """Generate filtering threshold recommendations based on statistics."""
    print("\n" + "="*80)
    print("FILTERING RECOMMENDATIONS")
    print("="*80)

    recommendations = []

    for _, row in stats_df.iterrows():
        segment = row['segment']

        # ONT-aware thresholds (preferred for ONT data)
        ont_threshold_2pct = row['ont_threshold_2pct']
        ont_threshold_5pct = row['ont_threshold_5pct']
        ont_threshold_10pct = row['ont_threshold_10pct']

        # Data-driven thresholds (95th and 99th percentiles)
        p95_threshold = row['p95']
        p99_threshold = row['p99']

        # Strict threshold (only perfect or near-perfect matches)
        strict_threshold = 1 if row['percent_ed1_or_less'] > 80 else 2

        rec = {
            'segment': segment,
            'median_length': row['median_length'],
            'ont_threshold_2pct': ont_threshold_2pct,
            'ont_threshold_5pct': ont_threshold_5pct,
            'ont_threshold_10pct': ont_threshold_10pct,
            'percent_passing_2pct': row['percent_passing_2pct'],
            'percent_passing_5pct': row['percent_passing_5pct'],
            'percent_passing_10pct': row['percent_passing_10pct'],
            'strict_threshold': strict_threshold,
            'p95_threshold': p95_threshold,
            'p99_threshold': p99_threshold,
            'percent_perfect': row['percent_perfect'],
        }

        recommendations.append(rec)

        print(f"\n{segment} (median length: {row['median_length']:.0f} bp):")
        print(f"  Perfect matches: {row['percent_perfect']:.1f}%")
        print(f"  Median edit distance: {row['median']:.1f}")
        print(f"\n  ONT-based thresholds (recommended for ONT data):")
        if not np.isnan(ont_threshold_2pct):
            print(f"    - Conservative (2% error):       edit_distance <= {ont_threshold_2pct} ({row['percent_passing_2pct']:.1f}% of reads)")
        if not np.isnan(ont_threshold_5pct):
            print(f"    - Permissive ({int(ont_error_rate*100)}% error):        edit_distance <= {ont_threshold_5pct} ({row['percent_passing_5pct']:.1f}% of reads)")
        if not np.isnan(ont_threshold_10pct):
            print(f"    - Very permissive (10% error):   edit_distance <= {ont_threshold_10pct} ({row['percent_passing_10pct']:.1f}% of reads)")

        print(f"\n  Data-driven thresholds:")
        print(f"    - Very strict (perfect only):       edit_distance <= {strict_threshold}")
        print(f"    - 95th percentile:                  edit_distance <= {p95_threshold:.0f}")
        print(f"    - 99th percentile:                  edit_distance <= {p99_threshold:.0f}")

    print("\n" + "="*80)
    print("RECOMMENDED FILTERING CODE (ONT-aware)")
    print("="*80)
    print("\nimport pandas as pd")
    print("df = pd.read_parquet('annotations_valid.parquet')")

    print("\n# Conservative filtering (2% error rate - high quality)")
    print("df_conservative = df[")
    for rec in recommendations:
        if not np.isnan(rec['ont_threshold_2pct']):
            print(f"    (df['{rec['segment']}_edit_distance'] <= {rec['ont_threshold_2pct']}) &")
    print("].copy()")

    print("\n# Permissive filtering (5% error rate - balance quality/quantity)")
    print("df_permissive = df[")
    for rec in recommendations:
        if not np.isnan(rec['ont_threshold_5pct']):
            print(f"    (df['{rec['segment']}_edit_distance'] <= {rec['ont_threshold_5pct']}) &")
    print("].copy()")

    print("\n# Very permissive filtering (10% error rate - maximum read retention)")
    print("df_very_permissive = df[")
    for rec in recommendations:
        if not np.isnan(rec['ont_threshold_10pct']):
            print(f"    (df['{rec['segment']}_edit_distance'] <= {rec['ont_threshold_10pct']}) &")
    print("].copy()")

    print("\n# Data-driven filtering (95th percentile)")
    print("df_p95 = df[")
    for rec in recommendations:
        print(f"    (df['{rec['segment']}_edit_distance'] <= {rec['p95_threshold']:.0f}) &")
    print("].copy()")

    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
ONT-based thresholds account for segment length:
  - Shorter segments (e.g., UMI ~12bp) tolerate fewer errors
  - Longer segments (e.g., p5 ~25bp) can have more errors while staying under error %
  - 2% error threshold: conservative, high-quality filtering
  - 5% error threshold: permissive, within typical ONT specs (2-5% error rate)
  - 10% error threshold: very permissive, for maximum read retention or lower quality runs

Data-driven thresholds (percentiles) show your actual data distribution:
  - Compare ONT thresholds vs percentiles to see if data matches expectations
  - If p95 >> ONT 5% threshold: data may have quality issues
  - If p95 ≈ ONT 5% threshold: data quality matches ONT expectations
  - If p95 << ONT 5% threshold: data quality is better than expected
""")

    return pd.DataFrame(recommendations)

def main():
    parser = argparse.ArgumentParser(
        description='Analyze edit distance distributions in annotated reads with ONT-specific thresholds'
    )
    parser.add_argument('parquet_file', help='Path to annotations parquet file')
    parser.add_argument('--output_dir', '-o',
                       help='Output directory for results (default: same as input file)',
                       default=None)
    parser.add_argument('--ont_error_rate', '-e', type=float, default=0.05,
                       help='Expected ONT error rate (default: 0.05 = 5%%)')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.parquet_file).parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_valid_reads(args.parquet_file)

    # Identify edit distance columns
    edit_dist_cols, barcode_segments, fixed_segments = identify_edit_distance_columns(df)
    all_segments = barcode_segments + fixed_segments

    if not edit_dist_cols:
        print("ERROR: No edit distance columns found in the parquet file!")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)

    print(f"\nFound edit distance columns for {len(all_segments)} segments:")
    print(f"  Barcode segments: {', '.join(barcode_segments)}")
    print(f"  Fixed segments: {', '.join(fixed_segments)}")

    # Calculate segment lengths
    print("\nCalculating median segment lengths...")
    segment_lengths = get_segment_lengths(df, all_segments)
    for segment, length in segment_lengths.items():
        print(f"  {segment}: {length:.0f} bp")

    # Calculate statistics for each segment
    print(f"\nCalculating statistics (ONT error rate: {args.ont_error_rate*100:.0f}%)...")
    stats_list = []
    for segment in all_segments:
        edit_dist_col = f'{segment}_edit_distance'
        segment_length = segment_lengths.get(segment)
        stats = calculate_statistics(df, segment, edit_dist_col,
                                     segment_length=segment_length,
                                     ont_error_rate=args.ont_error_rate)
        if stats:
            stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    # Save statistics to TSV
    stats_file = output_dir / 'edit_distance_statistics.tsv'
    stats_df.to_csv(stats_file, sep='\t', index=False, float_format='%.2f')
    print(f"\nSaved statistics to {stats_file}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    display_cols = ['segment', 'median_length', 'reads_with_segment', 'percent_perfect',
                    'median', 'ont_threshold_2pct', 'percent_passing_2pct',
                    'ont_threshold_5pct', 'percent_passing_5pct',
                    'ont_threshold_10pct', 'percent_passing_10pct']
    print(stats_df[display_cols].to_string(index=False))

    # Create visualizations
    plot_file = output_dir / 'edit_distance_distributions.pdf'
    plot_edit_distance_distributions(df, all_segments, segment_lengths,
                                     args.ont_error_rate, plot_file)
    print(f"\nSaved visualizations to {plot_file}")

    # Generate filtering recommendations
    recommendations_df = generate_filtering_recommendations(stats_df, args.ont_error_rate)
    rec_file = output_dir / 'filtering_recommendations.tsv'
    recommendations_df.to_csv(rec_file, sep='\t', index=False, float_format='%.1f')
    print(f"\nSaved recommendations to {rec_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
