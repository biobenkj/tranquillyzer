import pysam
import multiprocessing
import os
import subprocess

def parse_header(header):
    """Parses the read header to extract cell barcodes and UMI."""
    parts = header.split("_")  # Read name structure: UUID_combinedBarcodes_UMI
    
    if len(parts) < 3:
        return None, None  # Return None if format is unexpected
    
    combined_cb = parts[1]  # Barcodes are in the second position
    umi = parts[2]  # UMI is in the third position
    
    return combined_cb, umi

def add_tags_in_chunk(chunk):
    """Processes a chunk of SAM file to add CB and UMI tags while collecting alignment stats in local memory."""
    processed_lines = []
    local_stats = {
        "Total Reads": 0,
        "Aligned Reads": 0,
        "Unaligned Reads": 0,
        "Multi-mapped Reads": 0,
        "Supplementary Alignments": 0
    }
    
    for line in chunk:
        if line.startswith("@"):  # Keep SAM headers unchanged
            processed_lines.append(line)
            continue
        
        fields = line.strip().split("\t")
        read_name = fields[0]
        
        combined_cb, umi = parse_header(read_name)
        
        if combined_cb:
            fields.append(f"CB:Z:{combined_cb}")
        if umi:
            fields.append(f"RX:Z:{umi}")
        
        processed_lines.append("\t".join(fields) + "\n")
        
        # Collect alignment stats in local dictionary
        local_stats["Total Reads"] += 1
        flag = int(fields[1])
        if flag & 0x4:
            local_stats["Unaligned Reads"] += 1
        else:
            local_stats["Aligned Reads"] += 1
            if flag & 0x800:
                local_stats["Supplementary Alignments"] += 1
            if any(f.startswith("NH:i:") for f in fields):
                nh_value = int([x.split(":")[-1] for x in fields if x.startswith("NH:i:")][0])
                if nh_value > 1:
                    local_stats["Multi-mapped Reads"] += 1
    
    return processed_lines, local_stats

def add_tags_parallel(input_sam, output_sam, threads, chunk_size=100000):
    """Parallel processing of SAM file using lazy loading while collecting stats efficiently."""
    global_stats = {
        "Total Reads": 0,
        "Aligned Reads": 0,
        "Unaligned Reads": 0,
        "Multi-mapped Reads": 0,
        "Supplementary Alignments": 0
    }

    with open(output_sam, "w") as sam_out, open(input_sam, "r") as sam_in:
        header_lines = []
        chunk = []
        pool = multiprocessing.Pool(threads)
        results = []

        for line in sam_in:
            if line.startswith("@"):  # Store header separately
                header_lines.append(line)
            else:
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    results.append(pool.apply_async(add_tags_in_chunk, (chunk,)))
                    chunk = []

        if chunk:
            results.append(pool.apply_async(add_tags_in_chunk, (chunk,)))

        pool.close()
        pool.join()

        # Merge results from all parallel processes
        for res in results:
            processed_lines, local_stats = res.get()
            sam_out.writelines(processed_lines)

            # Merge local stats into global stats
            for key in global_stats:
                global_stats[key] += local_stats[key]

        sam_out.seek(0, 0)
        sam_out.writelines(header_lines)

    print(f"Processed SAM file saved as {output_sam}")
    return global_stats