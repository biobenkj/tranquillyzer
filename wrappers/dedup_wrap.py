def dedup_wrap(input_dir, lv_threshold, stranded, per_cell, threads):
    import os
    import time
    import resource
    import pysam
    import subprocess
    import logging
    from scripts.deduplicate import deduplication_parallel

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    start = time.time()
    logger.info("Starting duplicate marking process")

    aligned_bam = "aligned_files/demuxed_aligned.bam"
    dup_marked_bam = "aligned_files/demuxed_aligned_dup_marked.bam"

    input_bam = os.path.join(input_dir, aligned_bam)
    out_bam = os.path.join(input_dir, dup_marked_bam)

    deduplication_parallel(
        input_bam, out_bam, lv_threshold, per_cell, threads, stranded
    )

    if not os.path.exists(out_bam):
        raise FileNotFoundError(f"Expected output BAM not found: {out_bam}")

    try:
        with pysam.AlignmentFile(out_bam, "rb") as bam:
            so = (bam.header.get("HD") or {}).get("SO")
    except Exception as e:
        logger.warning(
            f"Could not read BAM header to check sort order ({e}). Assuming unsorted."
        )
        so = None

    logger.info("Indexing duplicate marked BAM file")
    subprocess.run(f"samtools index -@ {threads} {out_bam}", shell=True, check=True)
    logger.info(f"Indexing completed for {out_bam}")

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_mb = (
        usage.ru_maxrss / 1024 if os.uname().sysname == "Linux" else usage.ru_maxrss
    )  # Linux gives KB
    logger.info(f"Peak memory usage: {max_rss_mb:.2f} MB")
    logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
