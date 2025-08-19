import subprocess
from pathlib import Path
import pytest

raw_input_dir = Path("tests/10x3p/data")
output_dir = Path("tests/10x3p")
barcodes = Path("tests/10x3p/barcodes.tsv")
reference_fasta = Path("tests/references/hg38_gencode_chr21.fa")
threads = 2

output_dir.mkdir(exist_ok=True, parents=True)


def run_cmd(cmd, timeout=600):
    print(f"\n>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=timeout)
    print(result.stdout.decode())
    print(result.stderr.decode())
    assert result.returncode == 0, f"Command failed: {' '.join(cmd)}"


@pytest.mark.order(1)
def test_preprocess():
    run_cmd(["tranquillyzer", "preprocess", str(raw_input_dir),
             str(output_dir), "--threads", str(threads)])


@pytest.mark.order(2)
def test_readlengthdist():
    run_cmd(["tranquillyzer", "readlengthdist", str(output_dir)])


@pytest.mark.order(3)
def test_annotate_reads():
    run_cmd(["tranquillyzer", "annotate-reads", "10x3p_sc_ont_011",
             str(output_dir), str(barcodes), "--model-type", "CRF",
             "--chunk-size", str(100000), "--threads", str(threads)])


@pytest.mark.order(4)
def test_align():
    run_cmd(["tranquillyzer", "align", str(output_dir), str(reference_fasta),
             str(output_dir), "--preset", "splice", "--threads", str(threads)])


@pytest.mark.order(5)
def test_dedup():
    run_cmd(["tranquillyzer", "dedup", str(output_dir), "--lv-threshold",
             str(1), "--threads", str(threads), "--per-cell"])
