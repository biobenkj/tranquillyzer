import pandas as pd
import os
import time
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import numpy as np
from Levenshtein import distance as levenshtein_distance
from multiprocessing import Pool, cpu_count

# Move the function to the global scope
def compute_distance(args):
    """Compute the Levenshtein distance between a pair of UMIs."""
    i, j, umi_list = args
    return (i, j, levenshtein_distance(umi_list[i], umi_list[j]))

def generate_umi_pairs(umi_list):
    """Generate all unique UMI pairs for comparison."""
    n = len(umi_list)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return pairs

def parallel_stringdistmatrix(umi_list, num_workers=None):
    """Compute pairwise Levenshtein distance matrix in parallel."""
    if num_workers is None:
        num_workers = cpu_count()
    
    n = len(umi_list)
    pairs = generate_umi_pairs(umi_list)
    
    # Parallel execution of distance calculation
    with Pool(processes=num_workers) as pool:
        results = pool.map(compute_distance, [(i, j, umi_list) for i, j in pairs])
    
    # Create an empty distance matrix
    distances = np.zeros((n, n))
    
    # Populate the distance matrix with the results
    for i, j, dist in results:
        distances[i, j] = dist
        distances[j, i] = dist  # The matrix is symmetric
    
    return distances

def correct_umis(umi_list, threshold=1, num_workers=None):
    """Correct UMIs using hierarchical clustering with a parallel computed distance matrix."""
    start_time = time.time()
    
    # Compute pairwise distances using the parallel stringdistmatrix function
    distances = parallel_stringdistmatrix(umi_list, num_workers)
    
    # Log time taken for distance calculation
    dist_calc_time = time.time() - start_time
    print(f"[INFO] Distance matrix calculation completed in {dist_calc_time:.2f} seconds.")
    
    # Convert the distance matrix to condensed form
    condensed_distances = squareform(distances)
    
    # Perform hierarchical clustering
    Z = linkage(condensed_distances, method='single')
    
    # Form clusters with a given distance threshold
    cluster_labels = fcluster(Z, threshold, criterion='distance')
    
    # Map each UMI to its most common representative in its cluster
    corrected_umis = {}
    for label in set(cluster_labels):
        cluster_indices = [i for i, x in enumerate(cluster_labels) if x == label]
        cluster_umis = [umi_list[i] for i in cluster_indices]
        most_frequent_umi = max(set(cluster_umis), key=cluster_umis.count)
        for umi in cluster_umis:
            corrected_umis[umi] = most_frequent_umi
    
    # Log total time taken for UMI correction
    total_time = time.time() - start_time
    print(f"[INFO] UMI correction completed in {total_time:.2f} seconds.")
    
    return corrected_umis

def process_and_correct_umis(input_file, output_file, umi_column='UMI_Sequences', threshold=1, num_workers=None):
    """Collects all UMIs, corrects them globally with parallelized distance calculation, and maps back to original sequences."""
    
    start_time = time.time()
    
    print(f"[INFO] Collecting UMIs from file: {input_file}")
    
    # Step 1: Collect all UMIs
    all_umis = pd.read_csv(input_file, sep='\t', usecols=[umi_column])
    umi_list = all_umis[umi_column].tolist()
    
    # Log time taken for loading UMIs
    load_time = time.time() - start_time
    print(f"[INFO] Loaded {len(umi_list)} UMIs in {load_time:.2f} seconds.")
    
    # Reset start time for correction
    start_time = time.time()
    
    print(f"[INFO] Correcting {len(umi_list)} UMIs globally with parallelized distance calculation...")
    
    # Step 2: Correct all UMIs together with parallelized distance calculation
    corrected_umis = correct_umis(umi_list, threshold, num_workers)
    
    # Log time taken for correction
    correction_time = time.time() - start_time
    print(f"[INFO] UMI correction completed in {correction_time:.2f} seconds.")
    
    # Step 3: Map corrected UMIs back to the original DataFrame
    all_umis['Corrected_UMI'] = all_umis[umi_column].map(corrected_umis)
    
    # Step 4: Write the results to the output file
    print(f"[INFO] Writing corrected UMIs to output file: {output_file}")
    all_umis.to_csv(output_file, sep='\t', index=False)
    
    # Log total time taken for the entire process
    total_time = time.time() - start_time
    print(f"[INFO] Processing complete. Output written to {output_file}. Total time: {total_time:.2f} seconds.")

# Example usage:
# input_file = 'input_file.tsv'
# output_file = 'output_file_corrected.tsv'
# process_and_correct_umis(input_file, output_file, umi_column='UMI_Sequences', threshold=2, num_workers=4)