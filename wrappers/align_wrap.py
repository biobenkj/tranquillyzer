def align_wrap(
    input_dir, ref, output_dir, preset, filt_flag, mapq, threads, add_minimap_args
):

    import os
    import time
    import subprocess
    import resource
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    start = time.time()

    if os.path.exists(os.path.join(input_dir, "demuxed_fasta/demuxed.fastq")):
        fasta_file = os.path.join(input_dir, "demuxed_fasta/demuxed.fastq")
    elif os.path.exists(os.path.join(input_dir, "demuxed_fasta/demuxed.fasta")):
        fasta_file = os.path.join(input_dir, "demuxed_fasta/demuxed.fasta")
    else:
        raise FileNotFoundError(
            "No demuxed FASTA or FASTQ file found in the input directory."
        )

    os.makedirs(f"{output_dir}/aligned_files", exist_ok=True)
    output_bam_dir = os.path.join(output_dir, "aligned_files")

    output_bam = os.path.join(output_bam_dir, "demuxed_aligned.bam")

    minimap2_cmd = f"minimap2 -t {threads} -ax {preset} {add_minimap_args} \
        {ref} {fasta_file} | samtools view -h -F {filt_flag} -q {mapq} \
            -@ {threads} | samtools sort -@ {threads} -o {output_bam}"

    logger.info("Aligning reads to the reference genome")
    subprocess.run(minimap2_cmd, shell=True, check=True)
    logger.info(f"Alignment completed and sorted BAM saved as {output_bam}")

    logger.info(f"Indexing {output_bam}")
    subprocess.run(f"samtools index -@ {threads} {output_bam}", shell=True, check=True)
    logger.info("Indexing complete")

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_mb = (
        usage.ru_maxrss / 1024 if os.uname().sysname == "Linux" else usage.ru_maxrss
    )  # Linux gives KB
    logger.info(f"Peak memory usage during alignment: {max_rss_mb:.2f} MB")
    logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
