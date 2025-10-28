#!/usr/bin/env python3
"""
Filter ACC priors to only include sequences matching IUPAC pattern.

Usage:
    python filter_acc_priors.py input.tsv output.tsv tranquil ACCSSV
    
This will:
1. Read input.tsv
2. Filter to sequences matching the IUPAC pattern
3. Renormalize frequencies to sum to 1.0
4. Write filtered results to output.tsv
"""

import sys
import pandas as pd

# IUPAC codes
IUPAC_CODES = {
    'A': ['A'],
    'C': ['C'],
    'G': ['G'],
    'T': ['T'],
    'S': ['G', 'C'],          # Strong
    'V': ['A', 'C', 'G'],     # Not T
    'W': ['A', 'T'],          # Weak
    'R': ['A', 'G'],          # Purine
    'Y': ['C', 'T'],          # Pyrimidine
    'M': ['A', 'C'],          # Amino
    'K': ['G', 'T'],          # Keto
    'H': ['A', 'C', 'T'],     # Not G
    'B': ['C', 'G', 'T'],     # Not A
    'D': ['A', 'G', 'T'],     # Not C
    'N': ['A', 'C', 'G', 'T'] # Any
}

def validate_sequence(seq, pattern):
    """Check if sequence matches IUPAC pattern."""
    if len(seq) != len(pattern):
        return False, f"Length mismatch"
    
    for i, (seq_base, pattern_base) in enumerate(zip(seq, pattern)):
        if pattern_base in IUPAC_CODES:
            if seq_base not in IUPAC_CODES[pattern_base]:
                return False, f"Position {i+1}: {seq_base} not in {pattern_base}"
        else:
            if seq_base != pattern_base:
                return False, f"Position {i+1}: mismatch"
    
    return True, "Valid"

def main():
    if len(sys.argv) != 5:
        print("Usage: python filter_acc_priors.py <input.tsv> <output.tsv> <model_name> <iupac_pattern>")
        print("Example: python filter_acc_priors.py acc_priors.tsv acc_priors_filtered.tsv tranquil ACCSSV")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_name = sys.argv[3]
    iupac_pattern = sys.argv[4]
    
    print(f"Filtering ACC priors for model '{model_name}'")
    print(f"IUPAC pattern: {iupac_pattern}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 70)
    
    # Load file
    df = pd.read_csv(input_file, sep='\t', comment='#')
    
    # Get all entries for this model
    model_df = df[df['model_name'] == model_name].copy()
    
    if model_df.empty:
        print(f"❌ No entries found for model '{model_name}'")
        sys.exit(1)
    
    print(f"Found {len(model_df)} total sequences for model '{model_name}'\n")
    
    # Filter to valid sequences
    valid_rows = []
    invalid_rows = []
    
    for idx, row in model_df.iterrows():
        seq = row['sequence']
        is_valid, reason = validate_sequence(seq, iupac_pattern)
        
        if is_valid:
            valid_rows.append(row)
            print(f"✅ {seq} - {row['frequency']:.6f} - KEEP")
        else:
            invalid_rows.append(row)
            print(f"❌ {seq} - {row['frequency']:.6f} - REMOVE ({reason})")
    
    # Create filtered dataframe
    if not valid_rows:
        print("\n❌ No valid sequences found!")
        sys.exit(1)
    
    filtered_df = pd.DataFrame(valid_rows)
    
    # Renormalize frequencies
    original_sum = filtered_df['frequency'].sum()
    filtered_df['frequency'] = filtered_df['frequency'] / original_sum
    
    # Update notes to indicate renormalization
    filtered_df['notes'] = filtered_df.apply(
        lambda row: f"Renormalized from {original_sum:.6f}; " + str(row['notes']),
        axis=1
    )
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Valid sequences:   {len(valid_rows)}")
    print(f"Invalid sequences: {len(invalid_rows)}")
    print(f"Original frequency sum: {original_sum:.6f}")
    print(f"New frequency sum: {filtered_df['frequency'].sum():.6f} (renormalized to 1.0)")
    
    if invalid_rows:
        invalid_df = pd.DataFrame(invalid_rows)
        invalid_freq_sum = invalid_df['frequency'].sum()
        print(f"Removed frequency: {invalid_freq_sum:.6f} ({invalid_freq_sum/original_sum*100:.2f}%)")
    
    # Write output
    # Preserve other models if they exist
    other_models_df = df[df['model_name'] != model_name]
    
    if not other_models_df.empty:
        final_df = pd.concat([other_models_df, filtered_df], ignore_index=True)
        print(f"\nPreserved {len(other_models_df)} entries for other models")
    else:
        final_df = filtered_df
    
    final_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n✅ Filtered priors written to: {output_file}")
    
    # Show top 10
    print("\nTop 10 sequences after filtering:")
    print("-" * 70)
    top10 = filtered_df.nlargest(10, 'frequency')
    for _, row in top10.iterrows():
        print(f"  {row['sequence']}\t{row['frequency']:.6f}\t({row['frequency']*100:.2f}%)")

if __name__ == "__main__":
    main()
