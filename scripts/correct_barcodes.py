# import pandas as pd
# import multiprocessing as mp
# import Levenshtein as lev
# from functools import partial
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# # Function to calculate the reverse complement of a barcode
# def reverse_complement(seq):
#     complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
#     return ''.join(complement.get(base, base) for base in reversed(seq))

# # Function to correct a single observed barcode based on the whitelist
# def correct_barcode(row, column_name, whitelist, threshold):
#     observed_barcode = row[column_name]
#     reverse_comp_barcode = reverse_complement(observed_barcode)
#     read_name = row['ReadName']

#     # Calculate Levenshtein distances for both the observed barcode and its reverse complement
#     distances = [(wl_barcode, lev.distance(observed_barcode, wl_barcode)) for wl_barcode in whitelist]
#     distances += [(wl_barcode, lev.distance(reverse_comp_barcode, wl_barcode)) for wl_barcode in whitelist]

#     # Find the minimum distance
#     min_distance = min([d[1] for d in distances])

#     # Find all barcodes with the minimum distance
#     closest_barcodes = [d[0] for d in distances if d[1] == min_distance]

#     # If the minimum distance is greater than the threshold and there is a single closest barcode, correct it
#     if min_distance > threshold:
#         if len(closest_barcodes) == 1:
#             return observed_barcode, closest_barcodes[0], min_distance, 1  # Single match above threshold, correct
#         else:
#             return observed_barcode, "NMF", min_distance, len(closest_barcodes)  # Multiple matches or no match

#     # Otherwise, return the corrected barcode(s) if within threshold
#     return observed_barcode, ','.join(closest_barcodes), min_distance, len(closest_barcodes)

# # Wrapper function to parallelize correction for each barcode column with progress reporting
# def process_barcodes(chunk, barcode_columns, whitelist_dict, threshold, num_cores):
#     # Create a pool of workers
#     pool = mp.Pool(processes=num_cores)

#     results = []
    
#     with tqdm(total=len(chunk), desc="Processing chunk", unit="barcodes", leave=False) as pbar:
#         for _, row in chunk.iterrows():
#             result = {'ReadName': row['ReadName']}
            
#             for barcode_column in barcode_columns:
#                 whitelist = whitelist_dict[barcode_column]
#                 partial_func = partial(correct_barcode, column_name=barcode_column + "_Sequences", 
#                                        whitelist=whitelist, threshold=threshold)
#                 corrected_barcode, corrected_seq, min_dist, count = partial_func(row)
                
#                 result[f'corrected_{barcode_column}'] = corrected_seq
#                 result[f'corrected_{barcode_column}_min_dist'] = min_dist
#                 result[f'corrected_{barcode_column}_counts_with_min_dist'] = count
            
#             results.append(result)
#             pbar.update(1)

#     # Close the pool
#     pool.close()
#     pool.join()

#     # Convert the results into a DataFrame for output
#     df_results = pd.DataFrame(results)
#     return df_results

# # Function to generate side-by-side bar plots for each barcode type and save to a single PDF
# def generate_side_by_side_plots_to_pdf(results_df, barcode_columns, pdf_filename="barcode_plots.pdf"):
#     with PdfPages(pdf_filename) as pdf:
#         for barcode_column in barcode_columns:
#             # Get the count and min_dist columns for the current barcode type
#             count_column = f'corrected_{barcode_column}_counts_with_min_dist'
#             min_dist_column = f'corrected_{barcode_column}_min_dist'

#             # Count the occurrences of each unique value in both columns
#             count_data = results_df[count_column].value_counts().sort_index()
#             min_dist_data = results_df[min_dist_column].value_counts().sort_index()

#             # Create a figure with two subplots (side by side)
#             fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            
#             # Bar plot for the number of barcode matches
#             axs[0].bar(count_data.index, count_data.values, color='skyblue')
#             axs[0].set_xlabel(f'Number of Matches', fontsize=12)
#             axs[0].set_ylabel('Frequency', fontsize=12)
#             axs[0].set_title(f'{barcode_column}', fontsize=14)
#             axs[0].set_xticks(range(int(min(count_data.index)), int(max(count_data.index)) + 1, 1))  # Spacing x-axis ticks by 1
            
#             # Bar plot for the minimum distance
#             axs[1].bar(min_dist_data.index, min_dist_data.values, color='lightgreen')
#             axs[1].set_xlabel(f'Minimum Distance', fontsize=12)
#             axs[1].set_ylabel('Frequency', fontsize=12)
#             axs[1].set_title(f'{barcode_column}', fontsize=14)
#             axs[1].set_xticks(range(int(min(min_dist_data.index)), int(max(min_dist_data.index)) + 1, 1))  # Spacing x-axis ticks by 1

#             # Adjust layout and save both plots to the PDF
#             plt.tight_layout()
#             pdf.savefig(fig)
#             plt.close()

#     print(f"Bar plots saved to {pdf_filename}.")

# # Main function to execute the process for each barcode column and save output
# def barcode_correction_pipeline(input_file, whitelist_file, output_dir, column_mapping, threshold, num_cores, chunksize=10000):
#     # Read the whitelist once and store the relevant columns in a dictionary after dropping duplicates

#     output_file = output_dir + "/corrected_barcodes.tsv"

#     whitelist_df = pd.read_csv(whitelist_file, sep='\t')
#     whitelist_dict = {input_column: whitelist_df[whitelist_column].dropna().unique().tolist() 
#                       for input_column, whitelist_column in column_mapping.items()}

#     # Initialize flag for header
#     first_chunk = True
#     results = []

#     # Loop through each chunk of input file
#     for chunk in pd.read_csv(input_file, sep='\t', chunksize=chunksize):
#         # Process the current chunk of barcodes in parallel
#         results_df = process_barcodes(chunk, list(column_mapping.keys()), whitelist_dict, threshold, num_cores)

#         # Store the results for final plotting
#         results.append(results_df)

#         # Write to the output file in append mode (write header only for the first chunk)
#         results_df.to_csv(output_file, sep='\t', index=False, mode='a', header=first_chunk)
#         first_chunk = False  # Disable header writing for subsequent chunks

#     # Concatenate all results for plotting
#     final_results_df = pd.concat(results, ignore_index=True)

#     # Generate side-by-side bar plots and save them to a single PDF
#     generate_side_by_side_plots_to_pdf(final_results_df, list(column_mapping.keys()), 
#                                        pdf_filename=output_dir +"/barcode_plots.pdf")

#     print(f"Barcode correction completed. Results saved to {output_file}")

import pandas as pd
import multiprocessing as mp
import Levenshtein as lev
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product
from collections import defaultdict
from scripts.demultiplex import assign_cell_id

# Function to calculate the reverse complement of a barcode
def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

# Function to correct a single observed barcode based on the whitelist
def correct_barcode(row, column_name, whitelist, threshold):
    observed_barcode = row[column_name]
    reverse_comp_barcode = reverse_complement(observed_barcode)
    read_name = row['ReadName']

    # Calculate Levenshtein distances for both the observed barcode and its reverse complement
    distances = [(wl_barcode, lev.distance(observed_barcode, wl_barcode)) for wl_barcode in whitelist]
    distances += [(wl_barcode, lev.distance(reverse_comp_barcode, wl_barcode)) for wl_barcode in whitelist]

    # Find the minimum distance
    min_distance = min([d[1] for d in distances])

    # Find all barcodes with the minimum distance
    closest_barcodes = [d[0] for d in distances if d[1] == min_distance]

    # If the minimum distance is greater than the threshold and there is a single closest barcode, correct it
    if min_distance > threshold:
        if len(closest_barcodes) == 1:
            return observed_barcode, closest_barcodes[0], min_distance, 1  # Single match above threshold, correct
        else:
            return observed_barcode, "NMF", min_distance, len(closest_barcodes)  # Multiple matches or no match

    # Otherwise, return the corrected barcode(s) if within threshold
    return observed_barcode, ','.join(closest_barcodes), min_distance, len(closest_barcodes)

# Wrapper function to parallelize correction for each barcode column with progress reporting
# def process_barcodes(chunk, barcode_columns, whitelist_dict, threshold, num_cores):
#     # Create a pool of workers
#     pool = mp.Pool(processes=num_cores)

#     results = []
#     cDNA_lengths = []
    
#     with tqdm(total=len(chunk), desc="Processing chunk", unit="barcodes", leave=False) as pbar:
#         for _, row in chunk.iterrows():
#             result = {'ReadName': row['ReadName']}
#             result['read_length'] = row['read_length']
#             result['read'] = row['read']

#             result['cDNA_Starts'] = row['cDNA_Starts']
#             result['cDNA_Ends'] = row['cDNA_Ends']

#             result['cDNA_length'] = int(row['cDNA_Ends']) - int(row['cDNA_Starts']) + 1
            
#             for barcode_column in barcode_columns:
#                 whitelist = whitelist_dict[barcode_column]
#                 partial_func = partial(correct_barcode, column_name=barcode_column + "_Sequences", 
#                                        whitelist=whitelist, threshold=threshold)
#                 corrected_barcode, corrected_seq, min_dist, count = partial_func(row)
                
#                 result[f'{barcode_column}_Starts'] = row[f'{barcode_column}_Starts']
#                 result[f'{barcode_column}_Ends'] = row[f'{barcode_column}_Ends']
#                 result[f'corrected_{barcode_column}'] = corrected_seq
#                 result[f'corrected_{barcode_column}_min_dist'] = min_dist
#                 result[f'corrected_{barcode_column}_counts_with_min_dist'] = count

#             result['UMI_Starts'] = row['UMI_Starts']
#             result['UMI_Ends'] = row['UMI_Ends']

#             result['polyA_Starts'] = row['polyA_Starts']
#             result['polyA_Ends'] = row['polyA_Ends']
#             result['polyA_lengths'] = int(row['polyA_Ends']) - int(row['polyA_Starts']) + 1

#             result['random_s_Starts'] = row['random_s_Starts']
#             result['random_s_Ends'] = row['random_s_Ends']
#             result['random_e_Starts'] = row['random_e_Starts']
#             result['random_e_Ends'] = row['random_e_Ends']

#             result['architecture'] = row['architecture']
#             result['orientation'] = row['orientation']

#             results.append(result)
#             cDNA_lengths.append(result['cDNA_length'])
#             pbar.update(1)

#     # Close the pool
#     pool.close()
#     pool.join()

#     # Convert the results into a DataFrame for output
#     df_results = pd.DataFrame(results)
#     return df_results, cDNA_lengths

import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
import os

# def process_row(row, barcode_columns, whitelist_dict, whitelist_df, threshold):
#     # Initialize result dictionary with all annotations
#     result = {
#         'ReadName': row['ReadName'],
#         'read_length': row['read_length'],
#         'read': row['read'],
#         'cDNA_Starts': row['cDNA_Starts'],
#         'cDNA_Ends': row['cDNA_Ends'],
#         'cDNA_length': int(row['cDNA_Ends']) - int(row['cDNA_Starts']) + 1,
#         'UMI_Starts': row['UMI_Starts'],
#         'UMI_Ends': row['UMI_Ends'],
#         'polyA_Starts': row['polyA_Starts'],
#         'polyA_Ends': row['polyA_Ends'],
#         'polyA_lengths': int(row['polyA_Ends']) - int(row['polyA_Starts']) + 1,
#         'random_s_Starts': row['random_s_Starts'],
#         'random_s_Ends': row['random_s_Ends'],
#         'random_e_Starts': row['random_e_Starts'],
#         'random_e_Ends': row['random_e_Ends']
#     }

#     # Barcode correction
#     for barcode_column in barcode_columns:
#         whitelist = whitelist_dict[barcode_column]
#         corrected_barcode, corrected_seq, min_dist, count = correct_barcode(
#             row, barcode_column + "_Sequences", whitelist, threshold
#         )
#         result[f'corrected_{barcode_column}'] = corrected_seq
#         result[f'corrected_{barcode_column}_min_dist'] = min_dist
#         result[f'corrected_{barcode_column}_counts_with_min_dist'] = count
#         result[f'{barcode_column}_Starts'] = row[f'{barcode_column}_Starts']
#         result[f'{barcode_column}_Ends'] = row[f'{barcode_column}_Ends']

    
#     result['architecture'] = row['architecture']
#     result['orientation'] = row['orientation']

#     # Assign cell ID using the `result` dictionary
#     cell_id, local_match_counts, local_cell_counts = assign_cell_id(result, whitelist_df)
#     result['cell_id'] = cell_id

#     # Return the result along with match type and cell ID counts for aggregation
#     return result, local_match_counts, local_cell_counts

# def bc_n_demultiplex(chunk, barcode_columns, whitelist_dict, whitelist_df, threshold, num_cores):
#     def create_args(row):
#         return row, barcode_columns, whitelist_dict, whitelist_df, threshold

#     # Prepare arguments for parallel processing
#     args = [create_args(row) for _, row in chunk.iterrows()]

#     # Process rows in parallel
#     with Pool(num_cores) as pool:
#         results = list(
#             tqdm(pool.starmap(process_row, args),
#                  total=len(chunk), desc="Processing rows", unit="rows", leave=False)
#         )

#     # Separate processed rows and counts
#     processed_results = [res[0] for res in results]
#     all_match_type_counts = [res[1] for res in results]
#     all_cell_id_counts = [res[2] for res in results]

#     # Combine results into a DataFrame
#     df_results = pd.DataFrame(processed_results)

#     # Aggregate match type and cell ID counts
#     match_type_counts = defaultdict(int)
#     cell_id_counts = defaultdict(int)
#     cDNA_lengths = []

#     for result, match_counts, cell_counts in zip(processed_results, all_match_type_counts, all_cell_id_counts):
#         for key, value in match_counts.items():
#             match_type_counts[key] += value
#         for key, value in cell_counts.items():
#             cell_id_counts[key] += value
#         cDNA_lengths.append(result['cDNA_length'])

#     return df_results, cDNA_lengths, match_type_counts, cell_id_counts

def write_read_to_fasta(cell_id, read_name, cDNA_sequence, corrected_barcodes, umi_sequence, output_dir):
    """
    Write a read to a FASTA file named by cell_id in the output directory.
    Creates a new file if it doesn't exist; appends to it otherwise.
    """
    fasta_dir = os.path.join(output_dir, "demuxed_fasta")

    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)

    cell_id_str = str(cell_id)
    corrected_barcodes_str = str(corrected_barcodes)
    umi_sequence_str = str(umi_sequence)

    fasta_file_path = os.path.join(fasta_dir, f"{cell_id_str}.fasta")
    header = f">{read_name}|cell_id:{cell_id_str}|Barcodes:{corrected_barcodes_str}|UMI:{umi_sequence_str}\n"
    
    with open(fasta_file_path, "a") as fasta_file:
        fasta_file.write(header)
        fasta_file.write(f"{cDNA_sequence}\n")

def process_row(row, barcode_columns, whitelist_dict, whitelist_df, threshold, output_dir):
    # Initialize result dictionary with all annotations
    result = {
        'ReadName': row['ReadName'],
        'read_length': row['read_length'],
        'read': row['read'],
        'cDNA_Starts': row['cDNA_Starts'],
        'cDNA_Ends': row['cDNA_Ends'],
        'cDNA_length': int(row['cDNA_Ends']) - int(row['cDNA_Starts']) + 1,
        'UMI_Starts': row['UMI_Starts'],
        'UMI_Ends': row['UMI_Ends'],
        'polyA_Starts': row['polyA_Starts'],
        'polyA_Ends': row['polyA_Ends'],
        'polyA_lengths': int(row['polyA_Ends']) - int(row['polyA_Starts']) + 1,
        'random_s_Starts': row['random_s_Starts'],
        'random_s_Ends': row['random_s_Ends'],
        'random_e_Starts': row['random_e_Starts'],
        'random_e_Ends': row['random_e_Ends']
    }

    # Barcode correction
    corrected_barcodes = []
    for barcode_column in barcode_columns:
        whitelist = whitelist_dict[barcode_column]
        corrected_barcode, corrected_seq, min_dist, count = correct_barcode(
            row, barcode_column + "_Sequences", whitelist, threshold
        )
        result[f'corrected_{barcode_column}'] = corrected_seq
        result[f'corrected_{barcode_column}_min_dist'] = min_dist
        result[f'corrected_{barcode_column}_counts_with_min_dist'] = count
        result[f'{barcode_column}_Starts'] = row[f'{barcode_column}_Starts']
        result[f'{barcode_column}_Ends'] = row[f'{barcode_column}_Ends']
        corrected_barcodes.append(f"{barcode_column}:{corrected_seq}")

    corrected_barcodes_str = ";".join(corrected_barcodes)
    result['architecture'] = row['architecture']
    result['orientation'] = row['orientation']

    # Assign cell ID using the `result` dictionary
    cell_id, local_match_counts, local_cell_counts = assign_cell_id(result, whitelist_df)
    result['cell_id'] = cell_id

    cDNA_sequence = row['read'][int(row['cDNA_Starts']):int(row['cDNA_Ends']) + 1]
    umi_sequence = row['read'][int(row['UMI_Starts']):int(row['UMI_Ends']) + 1]

    # Save the read to a FASTA file named by cell_id
    write_read_to_fasta(cell_id, result['ReadName'], cDNA_sequence, corrected_barcodes_str, umi_sequence, output_dir)

    # Return the result along with match type and cell ID counts for aggregation
    return result, local_match_counts, local_cell_counts


def bc_n_demultiplex(chunk, barcode_columns, whitelist_dict, whitelist_df, threshold, output_dir, num_cores):
    def create_args(row):
        return row, barcode_columns, whitelist_dict, whitelist_df, threshold, output_dir

    # Prepare arguments for parallel processing
    args = [create_args(row) for _, row in chunk.iterrows()]

    # Process rows in parallel
    with Pool(num_cores) as pool:
        results = list(
            tqdm(pool.starmap(process_row, args),
                 total=len(chunk), desc="Processing rows", unit="rows", leave=False)
        )

    # Separate processed rows and counts
    processed_results = [res[0] for res in results]
    all_match_type_counts = [res[1] for res in results]
    all_cell_id_counts = [res[2] for res in results]

    # Combine results into a DataFrame
    df_results = pd.DataFrame(processed_results)

    # Aggregate match type and cell ID counts
    match_type_counts = defaultdict(int)
    cell_id_counts = defaultdict(int)
    cDNA_lengths = []

    for result, match_counts, cell_counts in zip(processed_results, all_match_type_counts, all_cell_id_counts):
        for key, value in match_counts.items():
            match_type_counts[key] += value
        for key, value in cell_counts.items():
            cell_id_counts[key] += value
        cDNA_lengths.append(result['cDNA_length'])

    return df_results, cDNA_lengths, match_type_counts, cell_id_counts

# Function to generate side-by-side bar plots for each barcode type and save to a single PDF
def generate_barcodes_stats_pdf(cumulative_barcodes_stats, barcode_columns, pdf_filename="barcode_plots.pdf"):
    with PdfPages(pdf_filename) as pdf:
        for barcode_column in barcode_columns:
            # Prepare data for plotting
            count_data = pd.Series(cumulative_barcodes_stats[barcode_column]['count_data']).sort_index()
            min_dist_data = pd.Series(cumulative_barcodes_stats[barcode_column]['min_dist_data']).sort_index()

            # Create a figure with two subplots (side by side)
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            
            # Bar plot for the number of barcode matches
            axs[0].bar(count_data.index, count_data.values, color='skyblue')
            axs[0].set_xlabel(f'Number of Matches', fontsize=12)
            axs[0].set_ylabel('Frequency', fontsize=12)
            axs[0].set_title(f'{barcode_column} - Number of Matches', fontsize=14)
            axs[0].set_xticks(range(int(min(count_data.index)), int(max(count_data.index)) + 1, 1))

            # Bar plot for the minimum distance
            axs[1].bar(min_dist_data.index, min_dist_data.values, color='lightgreen')
            axs[1].set_xlabel(f'Minimum Distance', fontsize=12)
            axs[1].set_ylabel('Frequency', fontsize=12)
            axs[1].set_title(f'{barcode_column} - Minimum Distance', fontsize=14)
            axs[1].set_xticks(range(int(min(min_dist_data.index)), int(max(min_dist_data.index)) + 1, 1))

            # Adjust layout and save both plots to the PDF
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    # with PdfPages(pdf_filename) as pdf:
    #     for barcode_column in barcode_columns:
    #         # Get the count and min_dist columns for the current barcode type
    #         count_column = f'corrected_{barcode_column}_counts_with_min_dist'
    #         min_dist_column = f'corrected_{barcode_column}_min_dist'

    #         # Count the occurrences of each unique value in both columns
    #         count_data = results_df[count_column].value_counts().sort_index()
    #         min_dist_data = results_df[min_dist_column].value_counts().sort_index()

    #         # Create a figure with two subplots (side by side)
    #         fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            
    #         # Bar plot for the number of barcode matches
    #         axs[0].bar(count_data.index, count_data.values, color='skyblue')
    #         axs[0].set_xlabel(f'Number of Matches', fontsize=12)
    #         axs[0].set_ylabel('Frequency', fontsize=12)
    #         axs[0].set_title(f'{barcode_column}', fontsize=14)
    #         axs[0].set_xticks(range(int(min(count_data.index)), int(max(count_data.index)) + 1, 1))  # Spacing x-axis ticks by 1
            
    #         # Bar plot for the minimum distance
    #         axs[1].bar(min_dist_data.index, min_dist_data.values, color='lightgreen')
    #         axs[1].set_xlabel(f'Minimum Distance', fontsize=12)
    #         axs[1].set_ylabel('Frequency', fontsize=12)
    #         axs[1].set_title(f'{barcode_column}', fontsize=14)
    #         axs[1].set_xticks(range(int(min(min_dist_data.index)), int(max(min_dist_data.index)) + 1, 1))  # Spacing x-axis ticks by 1

    #         # Adjust layout and save both plots to the PDF
    #         plt.tight_layout()
    #         pdf.savefig(fig)
    #         plt.close()

    print(f"Bar plots saved to {pdf_filename}.")

# Main function to execute the process for each barcode column and save output
def barcode_correction_pipeline(input_file, whitelist_file, output_dir, column_mapping, threshold, num_cores, chunksize=10000):
    # Read the whitelist once and store the relevant columns in a dictionary after dropping duplicates

    output_file = output_dir + "/corrected_barcodes.tsv"

    whitelist_df = pd.read_csv(whitelist_file, sep='\t')
    whitelist_dict = {input_column: whitelist_df[whitelist_column].dropna().unique().tolist() 
                      for input_column, whitelist_column in column_mapping.items()}

    # Initialize flag for header
    first_chunk = True
    results = []

    # Loop through each chunk of input file
    for chunk in pd.read_csv(input_file, sep='\t', chunksize=chunksize):
        # Process the current chunk of barcodes in parallel
        results_df = process_barcodes(chunk, list(column_mapping.keys()), whitelist_dict, threshold, num_cores)

        # Store the results for final plotting
        results.append(results_df)

        # Write to the output file in append mode (write header only for the first chunk)
        results_df.to_csv(output_file, sep='\t', index=False, mode='a', header=first_chunk)
        first_chunk = False  # Disable header writing for subsequent chunks

    # Concatenate all results for plotting
    final_results_df = pd.concat(results, ignore_index=True)

    # Generate side-by-side bar plots and save them to a single PDF
    generate_barcodes_stats_pdf(final_results_df, list(column_mapping.keys()),
                                pdf_filename=output_dir +"/barcode_plots.pdf")

    print(f"Barcode correction completed. Results saved to {output_file}")