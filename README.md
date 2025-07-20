# Tranquillyzer

Tranquillyzer is a Deep Learning (DL) based tool to annotate, visualize the annotated reads, demultiplex for single-cell long-reads data including **TRAQNUIL-seq** and **scNanoRNASeq** and generate some inital QC plots. 

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

To enhance the efficiency of the annotation process, tranquillyzer organizes raw reads into separate .parquet files, grouping them based on their lengths. This approach optimizes data compression within each bin, accelerates the annotation of the entire dataset, and facilitates the visualization of user-specified annotated reads without dependence on the annotation status of the complete dataset.

Example usage:

```console
tranquillyzer preprocessfasta /path/to/RAW_DATA/directory /path/to/OUTPUT/directory CPU_THREADS
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

**Note**: Before running the annotate-reads command, ensure you select the appropriate model for your dataset. If unsure, use the command `tranquillyzer availablemodels` to view the available models.

Example usage:

```console
tranquillyzer annotate-reads MODEL_NAME /path/to/OUTPUT/directory /path/to/BARCODE_WHITELIST --chunk-size 100000 --njobs @CPU_threads
```

### <ins>Read visualization</ins>

Annotated reads can be inspected independently of the `annotate-reads` process—either before or after successfully running the `annotate-reads` command—by providing their names to the `visualize` command. The resulting visualization is saved as a .pdf file in the **/path/to/OUTPUT/directory/plots/** folder.

Example usage:

```console
tranquillyzer visualize MODEL_NAME /path/to/OUTPUT/directory --read-names READ_NAME_1,READ_NAME_2,READ_NAME3
``` 



** More detailed usage instructions coming soon **
