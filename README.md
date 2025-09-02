# Tranquillyzer

[![codecov](https://codecov.io/github/AyushSemwal/tranquillyzer/graph/badge.svg?token=QS4IK3UZRN)](https://codecov.io/github/AyushSemwal/tranquillyzer)

**Tranquillyzer** (**TRAN**script **QU**antification **I**n **L**ong reads-ana**LYZER**), is a flexible, architecture-aware deep learning framework for processing long-read single-cell RNA-seq data. It employs a hybrid neural network architecture and a global, context-aware design, and enables precise identification of structural elements. In addition to supporting established single-cell protocols, Tranquillyzer accommodates custom library formats through rapid, one-time model training on user-defined label schemas, typically completed within a few hours on standard GPUs.

For a detailed description of the framework, benchmarking results, and application to real datasets, please refer to the [preprint](https://www.biorxiv.org/content/10.1101/2025.07.25.666829v1).

## Overview

### Preprocessing

Tranquillyzer implements a length-aware binning strategy that partitions reads into discrete, size-based bins (e.g., 0-499 bp, 500-999 bp, and so on). Each bin is written to a separate Parquet file, and binning is performed in parallel across multiple CPU threads to maximize preprocessing throughput. This strategy ensures that reads of similar lengths are grouped together, minimizing unnecessary padding and optimizing GPU memory consumption. In parallel, Tranquillyzer generates a lightweight index, mapping each read to its corresponding bin. This index enables rapid retrieval of individual reads for targeted visualization or debugging via the visualize sub-command, without reloading the full dataset.

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

Before proceeding with the installation, download the model files from this [link](https://www.dropbox.com/scl/fo/3lms8n97bnufzqa4ausv9/AGkO3EVrL1ZgctwEwTK1mEA?rlkey=47m69a6smwsisdznbwu3jvjpu&st=253al7s3&dl=0) and manually place the required model files inside the models/ directory:

For REG model, the following files are required:
* <model_name>.h5
* <model_name>_lbl_bin.pkl

Whereas for CRF model, the following files are required:
* <model_name>_w_CRF.h5
* <model_name>_w_CRF_lbl_bin.pkl
* <model_name>_w_CRF_params.json


These files are needed for annotation and visualization functionality.

### 3. <ins>Create and activate the environment</ins>

```bash
mamba env create -f environment.yml
conda activate tranquillyzer
```

### 4. <ins>Install GPU-enabled TensorFlow 2.15 with CUDA support</ins>

This ensures compatibility with your GPU and avoids system-wide CUDA requirements:

```bash
pip install "tensorflow[and-cuda]==2.15.1" --extra-index-url https://pypi.nvidia.com
```

Then install TensorFlow Addons:

```bash
pip install "tensorflow-addons==0.22.*"
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

The directory storing all the raw reads in **fasta/fa/fasta.gz/fa.gz/fastq/fq/fastq.gz/fq.gz** is provided as the input and the tool generates demultiplexed fasta files along with valid and invalid annotated **.parquet** files and some QC .pdf files as the outputs in two separate steps/commands (preprocessfasta and annotate-reads).

## Usage

### <ins>Preprocessing</ins>

To enhance the efficiency of the annotation process, tranquillyzer organizes raw reads into separate .parquet files, grouping them based on their lengths. This approach optimizes data compression within each bin, accelerates the annotation of the entire dataset, and facilitates the visualization of user-specified annotated reads without dependence on the annotation status of the complete dataset. Parallelization benefits are maximized when input data is provided as multiple files, as Tranquillyzer can distribute preprocessing and annotation tasks more effectively across CPU and GPU resources. Therefore, if raw data for a sample is available in separate files, we recommend preserving that structure rather than combining them into a single large file.

Example usage:

```console
tranquillyzer preprocess /path/to/RAW_DATA/directory /path/to/OUTPUT/directory \
    --threads CPU_THREADS
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

To select a model type (REG, CRF, or HYB), simply specify the base model name. Tranquillyzer will automatically detect the presence of the corresponding _w_CRF model file for CRF or HYB modes.

Model naming conventions:
	•	The REG model uses the base name (e.g., 10x3p_sc_ont_011.h5)
	•	The CRF model includes _w_CRF in its name (e.g., 10x3p_sc_ont_011_w_CRF.h5)

If only one version (REG or CRF) is available for a model, the user must select the corresponding model type explicitly. We recommend verifying which model versions are present before running annotate-reads.

Currently available trained models can be downloaded from this [link](https://www.dropbox.com/scl/fo/3lms8n97bnufzqa4ausv9/AGkO3EVrL1ZgctwEwTK1mEA?rlkey=47m69a6smwsisdznbwu3jvjpu&st=253al7s3&dl=0) and should be placed in the models/ folder within the cloned Tranquillyzer repository.

The command `tranquillyzer availablemodels` displays available models in the models/ folder within the cloned Tranquillyzer repository. The read architecture and exact sequences used to train the model are also shown with availablemodels.

Example usage:

```console
tranquillyzer annotate-reads /path/to/OUTPUT/directory /path/to/BARCODE_WHITELIST \
    --model-name MODEL_NAME --model-type CRF \
    --chunk-size 100000 --threads CPU_THREADS
```

### <ins>Alignment</ins>

```console
tranquilizer align INPUT_DIR REFERENCE OUTPUT_DIR \
    --preset MINIMAP2_PRESET --threads CPU_THREADS
```

### <ins>Duplicate Marking</ins>

```console
tranquilizer dedup INPUT_DIR \
    --lv-threshold EDIT_DISTANCE_THRESHOLD \
    --threads CPU_THREADS
```

### <ins>Read visualization</ins>

Annotated reads can be inspected independently of the `annotate-reads` process—either before or after successfully running the `annotate-reads` command—by providing their names to the `visualize` command. The resulting visualization is saved as a .pdf file in the **/path/to/OUTPUT/directory/plots/** folder.

Example usage:

```console
tranquillyzer visualize /path/to/OUTPUT/directory \
    --output-file OUTPUT_FILE_NAME \
    --model-name MODEL_NAME --model-type CRF \
    --read-names READ_NAME_1,READ_NAME_2,READ_NAME3 \
    --threads 2
```
```console
tranquillyzer visualize /path/to/OUTPUT/directory \
    --output-file OUTPUT_FILE_NAME \
    --model-name MODEL_NAME --model-type CRF \
    --num-reads 10 --threads 2
```

### <ins>Read visualization</ins>

Annotated reads can be inspected independently of the `annotate-reads` process—either before or after successfully running the `annotate-reads` command—by providing their names to the `visualize` command. The resulting visualization is saved as a .pdf file in the **/path/to/OUTPUT/directory/plots/** folder. If users want specific reads to be visualized, they can specify their names under the --read-names parameter. Otherwise, they can use the --num-reads parameter to visualize a randomly selected subset of reads from the entire dataset.

Example usage:

```console
tranquillyzer visualize /path/to/OUTPUT/directory \
    --output-file OUTPUT_FILE_NAME \
    --model-name MODEL_NAME --model-type CRF \
    --read-names READ_NAME_1,READ_NAME_2,READ_NAME3 \
    --threads 2
```
```console
tranquillyzer visualize /path/to/OUTPUT/directory \
    --output-file OUTPUT_FILE_NAME \
    --model-name MODEL_NAME --model-type CRF \
    --num-reads 10 --threads 2
```

** More detailed usage instructions coming soon **
