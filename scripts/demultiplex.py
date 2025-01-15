# import pandas as pd
# from multiprocessing import Pool, cpu_count, Manager
# from itertools import product
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from matplotlib.backends.backend_pdf import PdfPages

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

# # Function to process a chunk of the DataFrame in parallel
# def process_chunk(chunk, whitelist_df):
#     results = chunk.apply(lambda row: assign_cell_id(row, whitelist_df), axis=1)
    
#     # Prepare to return updated chunk and counters
#     chunk['cell_id'] = results.apply(lambda x: x[0])  # Get cell_id from result
#     match_type_counts = defaultdict(int)
#     cell_id_counts = defaultdict(int)
    
#     # Update local dictionaries with the returned values
#     for res in results:
#         for key, value in res[1].items():
#             match_type_counts[key] += value
#         for key, value in res[2].items():
#             cell_id_counts[key] += value
            
#     return chunk, match_type_counts, cell_id_counts

# # Parallel execution using multiprocessing
# def parallelize_chunk(chunk, whitelist_df, num_cores=None):
#     if num_cores is None:
#         num_cores = cpu_count()  # Use all available CPU cores
    
#     # Split the chunk into sub-chunks to process in parallel
#     chunk_size = len(chunk) // num_cores
#     sub_chunks = [chunk.iloc[i:i + chunk_size] for i in range(0, len(chunk), chunk_size)]
    
#     # Create a Pool of workers equal to the number of cores
#     with Pool(num_cores) as pool:
#         results = pool.starmap(process_chunk, [(sub_chunk, whitelist_df) for sub_chunk in sub_chunks])
    
#     # Combine the processed sub-chunks back into a single DataFrame
#     updated_chunk = pd.concat([res[0] for res in results])
    
#     # Combine all match_type and cell_id counters
#     total_match_type_counts = defaultdict(int)
#     total_cell_id_counts = defaultdict(int)
    
#     for res in results:
#         for key, value in res[1].items():
#             total_match_type_counts[key] += value
#         for key, value in res[2].items():
#             total_cell_id_counts[key] += value
    
#     return updated_chunk, total_match_type_counts, total_cell_id_counts

# # Main function to read file in chunks, process in parallel, and write to output
# def assign_cell_id_in_chunks(input_file, whitelist_file, output_dir, chunk_size, num_cores=None):

#     output_file = output_dir + "/demultiplexed_reads.tsv"
#     pdf_output_file = output_dir + "/demultiplexing_plots.pdf"

#     whitelist_df = pd.read_csv(whitelist_file, sep="\t", usecols=['i7', 'i5', 'CBC'])
    
#     # Initialize the main counters
#     match_type_counter = defaultdict(int)
#     cell_id_counter = defaultdict(int)

#     # Read the input file in chunks
#     with pd.read_csv(input_file, sep="\t", chunksize=chunk_size) as reader:
#         for i, chunk in enumerate(reader):
#             print(f"Processing chunk {i + 1}...")
#             processed_chunk, match_type_counts, cell_id_counts = parallelize_chunk(chunk, whitelist_df, num_cores)
            
#             # Update global counters after each chunk
#             for key, value in match_type_counts.items():
#                 match_type_counter[key] += value
#             for key, value in cell_id_counts.items():
#                 cell_id_counter[key] += value
            
#             # Write the processed chunk to the output file
#             if i == 0:
#                 # Write header only for the first chunk
#                 processed_chunk.to_csv(output_file, sep="\t", index=False, mode='w')
#             else:
#                 # Append without header for subsequent chunks
#                 processed_chunk.to_csv(output_file, sep="\t", index=False, mode='a', header=False)

#     print(f"Processing complete. Output saved to {output_file}")

#     # Generate histograms after processing
#     generate_histograms(pdf_output_file, match_type_counter, cell_id_counter)

# # Function to generate histograms and save them to a single PDF
# def generate_histograms(pdf_output_file, match_type_counter, cell_id_counter):
#     with PdfPages(pdf_output_file) as pdf:
#         # Predefined order for match categories
#         predefined_order = ['Exact match (i5 + i7 + CBC)', 'Two out of three match', 'Only CBC match', 'Ambiguous']
        
#         # Ensure match categories are plotted in predefined order
#         match_categories = [category for category in predefined_order if category in match_type_counter]
#         match_counts = [match_type_counter[category] for category in match_categories]

#         # Plot for match categories
#         if match_categories:
#             plt.figure(figsize=(8, 6))
#             plt.bar(match_categories, match_counts)
#             plt.title('Number of Reads per Match Category')
#             plt.xlabel('Match Category')
#             plt.ylabel('Number of Reads')
#             plt.xticks(rotation=45, ha='right')
#             plt.tight_layout()
#             pdf.savefig()  # Save this figure to the PDF
#             plt.close()
#         else:
#             print("No match types to plot.")

#         # Sort cell IDs by numeric order and plot
#         if cell_id_counter:
#             cell_id_data = {int(k): v for k, v in cell_id_counter.items() if k != "ambiguous"}
#             sorted_cell_ids = sorted(cell_id_data.keys())  # Sort numerically
#             read_counts = [cell_id_data[cell_id] for cell_id in sorted_cell_ids]
            
#             if len(sorted_cell_ids) > 50:
#                 # Limit the number of cell IDs shown to avoid cluttering the plot
#                 sorted_cell_ids = sorted_cell_ids[:50]
#                 read_counts = read_counts[:50]
                
#             plt.figure(figsize=(10, 6))
#             plt.bar([str(cell_id) for cell_id in sorted_cell_ids], read_counts)
#             plt.title('Number of Reads Assigned to Each Cell ID')
#             plt.xlabel('Cell ID')
#             plt.ylabel('Number of Reads')
#             plt.xticks(rotation=90, ha='right')  # Rotate cell IDs for better readability
#             plt.tight_layout()
#             pdf.savefig()  # Save this figure to the PDF
#             plt.close()
#         else:
#             print("No cell IDs to plot.")


import pandas as pd
from multiprocessing import Pool, cpu_count, Manager
from itertools import product
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

# Function to assign cell ID based on all possible combinations of i5, i7, and CBC
def assign_cell_id(row, whitelist_df):
    # Split the sequences by commas (if multiple) and generate all combinations
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

# Function to process a chunk of the DataFrame in parallel
def process_chunk(chunk, whitelist_df):
    results = chunk.apply(lambda row: assign_cell_id(row, whitelist_df), axis=1)
    
    # Prepare to return updated chunk and counters
    chunk['cell_id'] = results.apply(lambda x: x[0])  # Get cell_id from result
    match_type_counts = defaultdict(int)
    cell_id_counts = defaultdict(int)
    
    # Update local dictionaries with the returned values
    for res in results:
        for key, value in res[1].items():
            match_type_counts[key] += value
        for key, value in res[2].items():
            cell_id_counts[key] += value
            
    return chunk, match_type_counts, cell_id_counts

# Parallel execution using multiprocessing
def parallelize_assign_cell_id(chunk, whitelist_df, num_cores=None):
    if num_cores is None:
        num_cores = cpu_count()  # Use all available CPU cores
    
    # Split the chunk into sub-chunks to process in parallel
    chunk_size = len(chunk) // num_cores
    sub_chunks = [chunk.iloc[i:i + chunk_size] for i in range(0, len(chunk), chunk_size)]
    
    # Create a Pool of workers equal to the number of cores
    with Pool(num_cores) as pool:
        results = pool.starmap(process_chunk, [(sub_chunk, whitelist_df) for sub_chunk in sub_chunks])
    
    # Combine the processed sub-chunks back into a single DataFrame
    updated_chunk = pd.concat([res[0] for res in results])
    
    # Combine all match_type and cell_id counters
    total_match_type_counts = defaultdict(int)
    total_cell_id_counts = defaultdict(int)
    
    for res in results:
        for key, value in res[1].items():
            total_match_type_counts[key] += value
        for key, value in res[2].items():
            total_cell_id_counts[key] += value
    
    return updated_chunk, total_match_type_counts, total_cell_id_counts

# Main function to read file in chunks, process in parallel, and write to output
def assign_cell_id_in_chunks(input_file, whitelist_file, output_dir, chunk_size, num_cores=None):

    output_file = output_dir + "/demultiplexed_reads.tsv"
    pdf_output_file = output_dir + "/demultiplexing_plots.pdf"

    whitelist_df = pd.read_csv(whitelist_file, sep="\t", usecols=['i7', 'i5', 'CBC'])
    
    # Initialize the main counters
    match_type_counter = defaultdict(int)
    cell_id_counter = defaultdict(int)

    # Read the input file in chunks
    all_read_lengths = []
    all_cDNA_lengths = []

    with pd.read_csv(input_file, sep="\t", chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i + 1}...")

            # Collect read lengths and cDNA lengths for plotting later
            all_read_lengths.extend(chunk['read_length'].tolist())
            all_cDNA_lengths.extend(chunk['cDNA_length'].tolist())

            processed_chunk, match_type_counts, cell_id_counts = parallelize_assign_cell_id(chunk, whitelist_df, num_cores)
            
            # Update global counters after each chunk
            for key, value in match_type_counts.items():
                match_type_counter[key] += value
            for key, value in cell_id_counts.items():
                cell_id_counter[key] += value
            
            # Write the processed chunk to the output file
            if i == 0:
                # Write header only for the first chunk
                processed_chunk.to_csv(output_file, sep="\t", index=False, mode='w')
            else:
                # Append without header for subsequent chunks
                processed_chunk.to_csv(output_file, sep="\t", index=False, mode='a', header=False)

    print(f"Processing complete. Output saved to {output_file}")

    # Generate histograms after processing
    generate_demux_stats_pdf(pdf_output_file, match_type_counter, cell_id_counter, all_read_lengths, all_cDNA_lengths)

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