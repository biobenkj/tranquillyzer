import pandas as pd
from multiprocessing import Pool, cpu_count, Manager
from itertools import product
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

# # Function to assign cell ID based on all possible combinations of i5, i7, and CBC
# def assign_cell_id(row, whitelist_df):
#     # Split the sequences by commas (if multiple) and generate all combinations
#     i7_seqs = row['corrected_i7'].split(',')
#     i5_seqs = row['corrected_i5'].split(',')
#     cbc_seqs = row['corrected_CBC'].split(',')

#     # Generate all possible combinations of i7, i5, and CBC sequences
#     combinations = list(product(i7_seqs, i5_seqs, cbc_seqs))

#     exact_matches = []
#     two_matches = []
#     one_match = []

#     # Test each combination against the whitelist
#     for i7, i5, cbc in combinations:
#         match_all = whitelist_df[
#             (whitelist_df['i7'] == i7) &
#             (whitelist_df['i5'] == i5) &
#             (whitelist_df['CBC'] == cbc)
#         ]
        
#         match_two = whitelist_df[
#             ((whitelist_df['i7'] == i7) & (whitelist_df['i5'] == i5)) |
#             ((whitelist_df['i7'] == i7) & (whitelist_df['CBC'] == cbc)) |
#             ((whitelist_df['i5'] == i5) & (whitelist_df['CBC'] == cbc))
#         ]
        
#         match_cbc_only = whitelist_df[whitelist_df['CBC'] == cbc]

#         if not match_all.empty:
#             exact_matches.append(match_all.index[0] + 1)  # Make 1-based
#         elif not match_two.empty:
#             two_matches.append(match_two.index[0] + 1)  # Make 1-based
#         elif not match_cbc_only.empty:
#             one_match.append(match_cbc_only.index[0] + 1)  # Make 1-based

#     # Prepare result and counters
#     match_type_counter = defaultdict(int)
#     cell_id_counter = defaultdict(int)

#     # Determine the result and update local counters
#     if len(exact_matches) == 1:
#         match_type_counter['Exact match (i5 + i7 + CBC)'] += 1
#         cell_id_counter[str(exact_matches[0])] += 1  # Convert to str
#         return exact_matches[0], match_type_counter, cell_id_counter  # Exact match of i5, i7, CBC
#     elif len(two_matches) == 1:
#         match_type_counter['Two out of three match'] += 1
#         cell_id_counter[str(two_matches[0])] += 1  # Convert to str
#         return two_matches[0], match_type_counter, cell_id_counter  # Two out of three match
#     elif len(one_match) == 1:
#         match_type_counter['Only CBC match'] += 1
#         cell_id_counter[str(one_match[0])] += 1  # Convert to str
#         return one_match[0], match_type_counter, cell_id_counter  # Only CBC matches
#     else:
#         match_type_counter['Ambiguous'] += 1
#         return "ambiguous", match_type_counter, cell_id_counter  # Multiple or no matches

def assign_cell_id(row, whitelist_df, barcode_columns):
    # Check if only one barcode type is provided
    if len(barcode_columns) == 1:
        barcode_type = barcode_columns[0]  # e.g., 'i7', 'i5', or 'CBC'
        corrected_sequences = row[f"corrected_{barcode_type}"].split(',')

        # Match against the whitelist
        matches = whitelist_df[whitelist_df[barcode_type].isin(corrected_sequences)]

        # Prepare result and counters
        match_type_counter = defaultdict(int)
        cell_id_counter = defaultdict(int)

        if len(matches) == 1:  # One exact match
            match_type_counter[f'Exact match ({barcode_type})'] += 1
            cell_id = matches.index[0] + 1  # 1-based indexing
            cell_id_counter[str(cell_id)] += 1
            return cell_id, match_type_counter, cell_id_counter
        elif len(matches) > 1:  # Multiple matches
            match_type_counter[f'Ambiguous match ({barcode_type})'] += 1
            return "ambiguous", match_type_counter, cell_id_counter
        else:  # No match
            match_type_counter[f'No match ({barcode_type})'] += 1
            return "ambiguous", match_type_counter, cell_id_counter

    else:
        # Original logic when multiple barcodes are present
        i7_seqs = row['corrected_i7'].split(',')
        i5_seqs = row['corrected_i5'].split(',')
        cbc_seqs = row['corrected_CBC'].split(',')

        # Generate all possible combinations of i7, i5, and CBC sequences
        combinations = list(product(i7_seqs, i5_seqs, cbc_seqs))

        exact_matches = []
        two_matches = []
        one_match = []

        # Test each combination against the whitelist
        for i7, i5, cbc in combinations:
            match_all = whitelist_df[
                (whitelist_df['i7'] == i7) &
                (whitelist_df['i5'] == i5) &
                (whitelist_df['CBC'] == cbc)
            ]

            match_two = whitelist_df[
                ((whitelist_df['i7'] == i7) & (whitelist_df['i5'] == i5)) |
                ((whitelist_df['i7'] == i7) & (whitelist_df['CBC'] == cbc)) |
                ((whitelist_df['i5'] == i5) & (whitelist_df['CBC'] == cbc))
            ]

            match_cbc_only = whitelist_df[whitelist_df['CBC'] == cbc]

            if not match_all.empty:
                exact_matches.append(match_all.index[0] + 1)  # Make 1-based
            elif not match_two.empty:
                two_matches.append(match_two.index[0] + 1)  # Make 1-based
            elif not match_cbc_only.empty:
                one_match.append(match_cbc_only.index[0] + 1)  # Make 1-based

        # Prepare result and counters
        match_type_counter = defaultdict(int)
        cell_id_counter = defaultdict(int)

        # Determine the result and update local counters
        if len(exact_matches) == 1:
            match_type_counter['Exact match (i5 + i7 + CBC)'] += 1
            cell_id_counter[str(exact_matches[0])] += 1  # Convert to str
            return exact_matches[0], match_type_counter, cell_id_counter  # Exact match of i5, i7, CBC
        elif len(two_matches) == 1:
            match_type_counter['Two out of three match'] += 1
            cell_id_counter[str(two_matches[0])] += 1  # Convert to str
            return two_matches[0], match_type_counter, cell_id_counter  # Two out of three match
        elif len(one_match) == 1:
            match_type_counter['Only CBC match'] += 1
            cell_id_counter[str(one_match[0])] += 1  # Convert to str
            return one_match[0], match_type_counter, cell_id_counter  # Only CBC matches
        else:
            match_type_counter['Ambiguous'] += 1
            return "ambiguous", match_type_counter, cell_id_counter  # Multiple or no matches

# Function to generate histograms and save them to a single PDF
def generate_demux_stats_pdf(pdf_output_file, match_type_counter, cell_id_counter, read_lengths, cDNA_lengths):
    with PdfPages(pdf_output_file) as pdf:
        # Predefined order for match categories
        predefined_order = ['Exact match (i5 + i7 + CBC)', 'Two out of three match', 'Only CBC match', 'Ambiguous']
        
        # Ensure match categories are plotted in predefined order
        match_categories = [category for category in predefined_order if category in match_type_counter]
        match_counts = [match_type_counter[category] for category in match_categories]

        # Plot for match categories
        if match_categories:
            plt.figure(figsize=(8, 6))
            plt.bar(match_categories, match_counts)
            plt.title('Number of Reads per Match Category')
            plt.xlabel('Match Category')
            plt.ylabel('Number of Reads')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            pdf.savefig()  # Save this figure to the PDF
            plt.close()
        else:
            print("No match types to plot.")

        # Sort cell IDs by numeric order and plot
        if cell_id_counter:
            cell_id_data = {int(k): v for k, v in cell_id_counter.items() if k != "ambiguous"}
            sorted_cell_ids = sorted(cell_id_data.keys())  # Sort numerically
            read_counts = [cell_id_data[cell_id] for cell_id in sorted_cell_ids]
            
            if len(sorted_cell_ids) > 50:
                # Limit the number of cell IDs shown to avoid cluttering the plot
                sorted_cell_ids = sorted_cell_ids[:50]
                read_counts = read_counts[:50]
                
            plt.figure(figsize=(10, 6))
            plt.bar([str(cell_id) for cell_id in sorted_cell_ids], read_counts)
            plt.title('Number of Reads Assigned to Each Cell ID')
            plt.xlabel('Cell ID')
            plt.ylabel('Number of Reads')
            plt.xticks(rotation=90, ha='right')  # Rotate cell IDs for better readability
            plt.tight_layout()
            pdf.savefig()  # Save this figure to the PDF
            plt.close()
        else:
            print("No cell IDs to plot.")
        
        # Plot read length distribution
        plt.figure(figsize=(8, 6))
        plt.hist(read_lengths, bins=100, color='blue', edgecolor='black')
        plt.title('Read Length Distribution')
        plt.xlabel('Read Length')
        plt.ylabel('Frequency')
        plt.tight_layout()
        pdf.savefig()  # Save this figure to the PDF
        plt.close()

        # Plot cDNA length distribution
        plt.figure(figsize=(8, 6))
        plt.hist(cDNA_lengths, bins=100, color='green', edgecolor='black')
        plt.title('cDNA Length Distribution')
        plt.xlabel('cDNA Length')
        plt.ylabel('Frequency')
        plt.tight_layout()
        pdf.savefig()  # Save this figure to the PDF
        plt.close()