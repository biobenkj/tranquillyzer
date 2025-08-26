import os, subprocess
from pathlib import Path
import pytest

RAW_INPUT_DIR = Path("tests/10x3p/data")
OUT_DIR = Path("tests/10x3p")
BARCODES = Path("tests/10x3p/barcodes.tsv")
REF_FASTA = Path("tests/references/hg38_gencode_chr21.fa")
THREADS = 2

OUT_DIR.mkdir(exist_ok=True, parents=True)

COVRC = str(Path(".coveragerc").resolve())
COVDATA = str(Path(".coverage").resolve())  # single shared data file


def run_cmd(cmd, timeout=900):
    env = os.environ.copy()
    env["COVERAGE_PROCESS_START"] = COVRC
    env["COVERAGE_FILE"] = COVDATA
    env.setdefault("PYTHONPATH", os.getcwd())

    print(f"\n>> Running: {' '.join(map(str, cmd))}")
    p = subprocess.run(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    print(p.stdout)
    print(p.stderr)
    assert p.returncode == 0, f"Command failed ({p.returncode}): {' '.join(map(str, cmd))}"


@pytest.mark.order(1)
def test_preprocess():
    run_cmd([
        "tranquillyzer",
        "preprocess",
        RAW_INPUT_DIR,
        OUT_DIR,
        "--threads", THREADS,
    ])


@pytest.mark.order(2)
def test_readlengthdist():
    run_cmd([
        "tranquillyzer",
        "readlengthdist",
        OUT_DIR,
    ])


@pytest.mark.order(3)
def test_annotate_reads():
    run_cmd([
        "tranquillyzer",
        "annotate-reads",
        OUT_DIR,
        BARCODES,
        "--model-type", "CRF",
        "--chunk-size", 100000,
        "--threads", THREADS,
    ])


@pytest.mark.order(4)
def test_align():
    run_cmd([
        "tranquillyzer",
        "align",
        OUT_DIR,
        REF_FASTA,
        OUT_DIR,
        "--preset", "splice",
        "--threads", THREADS,
    ])


@pytest.mark.order(5)
def test_dedup():
    run_cmd([
        "tranquillyzer",
        "dedup",
        OUT_DIR,
        "--lv-threshold", 1,
        "--threads", THREADS,
        "--per-cell",
    ])
