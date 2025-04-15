import pysam
import multiprocessing as mp
import pandas as pd
import os
from collections import defaultdict
from rapidfuzz.distance import Levenshtein

# Extract Cell Barcode (CB) and UMI from read name
def extract_cb_umi(read_name):
    """Extracts cell barcode and UMI from the read name."""
    parts = read_name.split("_")
    if len(parts) >= 3:
        return parts[1], parts[2], "_".join(parts[:1] + parts[3:])  # Remove CBC & UMI from name
    return None, None, read_name

# Process reads for a genomic region
def process_region(bam_file, output_bam, region, umi_ld, per_cell, stranded):
    """Processes a single genomic region, deduplicates reads per cell if enabled, and writes to a temporary BAM file."""
    temp_bam_file = f"{output_bam}_{region}.bam"
    
    with pysam.AlignmentFile(bam_file, "rb") as bam_in, pysam.AlignmentFile(temp_bam_file, "wb", template=bam_in) as bam_out:
        reads = []
        for read in bam_in.fetch(region):
            if read.is_unmapped:
                continue  
            
            read_name = read.query_name
            cb, umi, clean_name = extract_cb_umi(read_name)
            chrom, start = read.reference_name, read.reference_start
            end = read.reference_end  # Capture end position
            if not cb or not umi:
                continue  
            
            strand = "-" if read.is_reverse else "+"
            read_data = (clean_name, read.flag, chrom, start, end, 
                         read.mapping_quality, read.cigarstring, read.seq, read.qual, strand, cb, umi)
            reads.append((cb, umi, start, end, read_data))
        
        marked_reads = mark_duplicates(reads, umi_ld, per_cell)
        
        for read_data in marked_reads:
            read_name, flag, chrom, start, end, mapping_quality, cigarstring, seq, qual, strand, cb, umi, duplicate = read_data
            read = pysam.AlignedSegment(bam_out.header)
            read.query_name = read_name  # Now without CBC and UMI
            read.flag = flag
            read.reference_name = chrom
            read.reference_start = start
            read.mapping_quality = mapping_quality
            read.cigarstring = cigarstring
            read.seq = seq
            read.qual = qual
            read.set_tag("CB", cb, value_type='Z')  # Cell Barcode Tag
            read.set_tag("UB", umi, value_type='Z')  # UMI Tag
            read.set_tag("DT", duplicate, value_type='Z')  # Duplicate Tag (Yes/No)

            # **Set PCR Duplicate Flag (0x400)**
            if duplicate == "Yes":
                read.flag |= 0x400  # Set duplicate flag
            else:
                read.flag &= ~0x400  # Ensure duplicate flag is unset for unique reads

            bam_out.write(read)

# Mark duplicate reads based on UMI similarity and position tolerance
def mark_duplicates(reads, umi_ld, per_cell, position_tolerance=10):
    """Marks duplicates while ensuring all reads are retained, supporting per-cell deduplication.
       Allows slight positional variations (default ±10bp)."""
    umi_clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    marked_reads = []
    
    for cb, umi, start, end, read_data in reads:
        key = cb if per_cell else "global"
        found_match = False
        
        # Allow grouping of reads within ±position_tolerance bp at both start and end
        for existing_start in umi_clusters[key]:
            if abs(existing_start - start) <= position_tolerance:
                for existing_end in umi_clusters[key][existing_start]:
                    if abs(existing_end - end) <= position_tolerance:
                        for existing_umi in umi_clusters[key][existing_start][existing_end]:
                            if Levenshtein.distance(umi, existing_umi) <= umi_ld:
                                umi_clusters[key][existing_start][existing_end][existing_umi].append((*read_data, "Yes"))
                                found_match = True
                                break
                    if found_match:
                        break
            if found_match:
                break
        
        if not found_match:
            umi_clusters[key][start][end][umi] = [(*read_data, "No")]
    
    for cell_umis in umi_clusters.values():
        for position_cluster in cell_umis.values():
            for end_cluster in position_cluster.values():
                for reads in end_cluster.values():
                    marked_reads.extend(reads)
    
    return marked_reads

# Merge BAM files from parallel processing
def merge_bam_files(output_bam, regions):
    """Merges temporary BAM files into a single output BAM."""
    temp_files = [f"{output_bam}_{region}.bam" for region in regions]
    merged_bam = output_bam
    
    with pysam.AlignmentFile(temp_files[0], "rb") as template:
        with pysam.AlignmentFile(merged_bam, "wb", template=template) as merged_out:
            for temp_file in temp_files:
                with pysam.AlignmentFile(temp_file, "rb") as temp_bam:
                    for read in temp_bam:
                        merged_out.write(read)
                os.remove(temp_file)  # Clean up temporary BAM files
    
    print(f"Merged BAM file saved as {merged_bam}")

# Compute and save deduplication statistics
def compute_final_stats(bam_file):
    """Computes final statistics based on duplicate tags in the BAM file."""
    unique_reads = set()
    duplicate_reads = set()
    
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for read in bam:
            read_name = read.query_name
            duplicate_tag = read.get_tag("DT") if read.has_tag("DT") else "No"
            
            if duplicate_tag == "Yes":
                duplicate_reads.add(read_name)
            else:
                unique_reads.add(read_name)
    
    stats = {"Unique Reads": len(unique_reads), "Duplicate Reads": len(duplicate_reads)}
    stats_file = bam_file.replace(".bam", "_stats.tsv")
    pd.DataFrame(stats.items(), columns=["Metric", "Value"]).to_csv(stats_file, sep="\t", index=False)
    print(f"Final stats saved as {stats_file}")

# Run deduplication in parallel
def deduplication_parallel(sorted_bam, output_bam, umi_ld, per_cell, threads, stranded):
    """Parallel BAM processing per genomic location, supporting per-cell deduplication."""
    with pysam.AlignmentFile(sorted_bam, "rb") as bam:
        regions = list(bam.references)
    
    pool = mp.Pool(threads)
    pool.starmap(process_region, [(sorted_bam, output_bam, region, umi_ld, per_cell, stranded) for region in regions])
    
    pool.close()
    pool.join()
    
    merge_bam_files(output_bam, regions)
    compute_final_stats(output_bam)
    print(f"Final BAM saved as {output_bam}")