# Tranquillyzer

Tranquillyzer is a Deep Learning (DL) based tool to annotate, visualize the annotated reads, demultiplex for single-cell long-reads data including **TRANQUIL-seq** and 10x 3' and 5' libraries and generate some inital QC plots. 

## Overview

### Preprocessing

Tranquillyzer implements a length-aware binning strategy that partitions reads into discrete, size-based bins (e.g., 0-499 bp, 500-999 bp, and so on) (Figure 1A). Each bin is written to a separate Parquet file, and binning is performed in parallel across multiple CPU threads to maximize preprocessing throughput. This strategy ensures that reads of similar lengths are grouped together, minimizing unnecessary padding and optimizing GPU memory consumption. In parallel, Tranquillyzer generates a lightweight index, mapping each read to its corresponding bin. This index enables rapid retrieval of individual reads for targeted visualization or debugging via the visualize sub-command, without reloading the full dataset.

### Read Annotation and Demultiplexing

Reads are processed in batches organized by length, as defined during preprocessing. Within each bin, the batch size for annotation inference is dynamically scaled based on the average read length to balance memory usage and throughput. Once batched and encoded, reads are passed through the trained model to infer base-wise label sequences, enabling identification of key structural components such as adapters, cell barcodes (CBCs), unique molecular identifiers (UMIs), cDNA regions, and polyA/T tails. 

Model inference is distributed across all available GPU cores using TensorFlow’s MirroredStrategy, enabling each batch to be processed concurrently across devices. As each batch completes inference, its predictions are offloaded to a pool of CPU threads, configured via a user-defined threads parameter, for postprocessing. This stage includes label decoding, structural validation, barcode correction, and demultiplexing. From the per-base annotations, contiguous regions are aggregated to identify structural components within each read, including adapters, CBCs, polyA/polyT tails, UMIs, and RNA inserts. The structural validity of each read is assessed by comparing the predicted element order against a protocol-specific label sequence defined in the tab-delimited text file. Reads that conform to the expected structure are marked as valid; those that do not are flagged as invalid.

For structurally valid reads, annotated barcodes are compared against a provided whitelist using the Levenshtein edit distance. Reads with a unique match within a user-defined threshold (default: ≤ 2) are assigned the corresponding barcode, while those that fail to match or yield multiple equally close matches are labeled as ambiguous. In parallel, RNA insert sequences from structurally valid reads with successfully assigned barcodes are written to a demuxed.fasta file, with the corrected CBC embedded in the FASTA header for compatibility with downstream alignment and quantification pipelines. Reads with a valid structural layout but no confidently assigned barcode are instead saved to an ambiguous.fasta file for further inspection or potential rescue.

### Duplicate Marking

The demultiplexed reads are first aligned to a user-specified reference genome using minimap2 with spliced alignment settings and unmapped reads are discarded before storing into a coordinate-sorted BAM file. Within each region, reads with identical strand orientation and cell barcode and similar start-end genomic positions are
compared for UMI similarity. These filters are configurable, allowing users to relax constraints depending on their experimental design or tolerance for false positives. Duplicates are annotated by setting standard SAM tags and flags for each identified read. Finally, BAM files generated per genomic region are merged to produce a fully duplicate-marked output file, which can be post-processed using tools such as samtools for downstream filtering and analysis.

### Visualization

Tranquillyzer offers detailed, color-coded visualizations of read annotations, where individual structural elements, such as adapter sequences, polyA/T tails, cell barcodes, UMIs, and cDNA regions, are distinctly labeled to enable intuitive exploration of per-read architecture.

## Tool Architecture

![Tranquillyzer Overview](docs/tranquillyzer_schema.png)
 
## Installation

We recommend using **`mamba`** for efficient environment setup and reproducibility. The following steps guide you through a clean GPU-enabled installation.

### 1. <ins>Clone the Repository</ins>

```bash
git clone https://github.com/huishenlab/tranquillyzer.git
cd tranquillyzer
```

### 2. <ins>Add Model Files</ins>

Before proceeding with the installation, manually place the required model files inside the models/ directory:
* <model_name>.h5
* <model_name>_lbl_bin.pkl

These files are needed for annotation and visualization functionality.

### 3. <ins>Create and activate the environment</ins>

```bash
mamba env create -f environment.yml
conda activate tranquillyzer
```

### 4. <ins>Install GPU-enabled TensorFlow 2.15 with CUDA 11.8 support</ins>

This ensures compatibility with your GPU and avoids system-wide CUDA requirements:

```bash
python -m pip install "tensorflow[and-cuda]==2.15" --extra-index-url https://pypi.nvidia.com
```

Then install TensorFlow Addons:

```bash
pip install tensorflow-addons
```

### 5. <ins>Install tranquillyzer (in editable mode)</ins>

From the root directory of this repository:

```bash
pip install -e .
```

#### Fix any missing dependencies

If you see errors or warnings after installing TensorFlow (especially related to `bx-python`, `sklearn` etc.), run the following:

```bash
pip install bx-python scikit-learn
```

## Quick Start

Verify that tranquillyzer is installed correctly:

```bash
tranquillyzer --help
```

You should see the CLI help message with available commands like:

- `availablemodels`
- `preprocessfasta`
- `annotate-reads`
- `visualize`
- `align`
- `dedup`
- `simulate-data`
- `train-model`

## I/O

The directory storing all the raw reads in **fasta/fa/fasta.gz/fa.gz/fastq/fq/fastq.gz/fq.gz** is provided as the input and the tool generates demultiplexed fasta files along with valid and invalid annotated **.parquet** files and some QC .pdf files as the outputs in two separate steps/commands (preprocessfasta and annotate-reads). Check out the examples drectory for both **TRAQNUIL-seq** and **scNanoRNASeq** datasets.

## Usage

### <ins>Preprocessing</ins>

To enhance the efficiency of the annotation process, tranquillyzer organizes raw reads into separate .parquet files, grouping them based on their lengths. This approach optimizes data compression within each bin, accelerates the annotation of the entire dataset, and facilitates the visualization of user-specified annotated reads without dependence on the annotation status of the complete dataset. Parallelization benefits are maximized when input data is provided as multiple files, as Tranquillyzer can distribute preprocessing and annotation tasks more effectively across CPU and GPU resources. Therefore, if raw data for a sample is available in separate files, we recommend preserving that structure rather than combining them into a single large file.

Example usage:

```console
tranquillyzer preprocessfasta /path/to/RAW_DATA/directory /path/to/OUTPUT/directory --threads {CPU_THREADS}
```
It is recommended that you follow the directory structure as in the exmples.

### <ins>Read length distribution</ins>

As an initial quality control metric, users may wish to visualize the read length distribution. The `readlengthdist` command facilitates this by generating a plot with log10-transformed read lengths on the x-axis and their corresponding frequencies on the y-axis. The output is provided in .png format in the **/path/to/OUTPUT/directory/plots/** folder.

Example uage:

```console
tranquillyzer readlengthdist /path/to/OUTPUT/directory
```

### <ins>Annotation, barcode correction and demultiplexing</ins>

Reads can be annotated, followed by barcode extraction, correction, and assignment to their respective cells (demultiplexing), using the single command `annotate-reads`. This command produces the following outputs:
* Demultiplexed FASTA files: Located in /path/to/OUTPUT/directory/demuxed_fasta/.
* Annotation metadata:
	1. Valid reads: /path/to/OUTPUT/directory/annotations_valid.parquet
    2. Invalid reads: /path/to/OUTPUT/directory/annotations_invalid.parquet
* Quality control (QC) plots:
    1. barcode_plots.pdf
    2. demux_plots.pdf
    3. full_read_annots.pdf
All QC plots are saved in /path/to/OUTPUT/directory/plots/.

**Note**: Before running the annotate-reads command, ensure you select the appropriate model and model type for your dataset. Tranquillyzer supports multiple model types:
* REG (standard CNN-LSTM)
* CRF (CNN-LSTM with a CRF layer for improved label consistency)
* HYB (hybrid mode, which runs REG first and reprocesses invalid reads with CRF).

If unsure, use the command `tranquillyzer availablemodels` to view the available models.

Example usage:

```console
tranquillyzer annotate-reads MODEL_NAME /path/to/OUTPUT/directory /path/to/BARCODE_WHITELIST --model-type CRF --chunk-size 100000 --threads @CPU_threads
```

### <ins>Read visualization</ins>

Annotated reads can be inspected independently of the `annotate-reads` process—either before or after successfully running the `annotate-reads` command—by providing their names to the `visualize` command. The resulting visualization is saved as a .pdf file in the **/path/to/OUTPUT/directory/plots/** folder.

Example usage:

```console
tranquillyzer visualize MODEL_NAME /path/to/OUTPUT/directory --read-names READ_NAME_1,READ_NAME_2,READ_NAME3
``` 



** More detailed usage instructions coming soon **
