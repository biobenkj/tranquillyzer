# Tranquillyzer

[![codecov](https://codecov.io/github/AyushSemwal/tranquillyzer/graph/badge.svg?token=QS4IK3UZRN)](https://codecov.io/github/AyushSemwal/tranquillyzer)

**Tranquillyzer** (**TRAN**script **QU**antification **I**n **L**ong reads-ana**LYZER**), is a flexible,
architecture-aware deep learning framework for processing long-read single-cell RNA-seq (scRNA-seq) data. It employs a
hybrid neural network architecture and a global, context-aware design that enables the precise identification of
structural elements. In addition to supporting established single-cell protocols, Tranquillyzer accommodates custom
library formats through rapid, one-time model training on user-defined label schemas. Model training for both
established and custom protocols can typically be completed within a few hours on standard GPUs.

For a detailed description of the framework, benchmarking results, and application to real datasets, please refer to the
[preprint](https://www.biorxiv.org/content/10.1101/2025.07.25.666829v1).

# Citation

### bioRxiv

```
Tranquillyzer: A Flexible Neural Network Framework for Structural Annotation and
Demultiplexing of Long-Read Transcriptomes. Ayush Semwal, Jacob Morrison, Ian
Beddows, Theron Palmer, Mary F. Majewski, H. Josh Jang, Benjamin K. Johnson, Hui
Shen. bioRxiv 2025.07.25.666829; doi: https://doi.org/10.1101/2025.07.25.666829.
```

# Overview

Tranquillyzer includes several steps to process reads from a raw basecalled FASTA/FASTQ file to a deduplicated BAM to
creating a feature counts matrix. First, Tranquillyzer preprocesses the reads to collect metadata on the reads and sort
them into bins of similar lengths to ease downstream processing. Next, Tranquillyzer annotates the reads using a hybrid
neural network architecture to identify each structural element in a read. It also demultiplexes reads to their
respective cells at this time. After annotating and demultiplexing, the reads are aligned and PCR duplicate marked. The
BAM output from this step can then be used to determine feature counts matrices. Tranquillyzer also provides a variety
of associated functionality including visualizing annotated reads and quality control metrics, training models for new
sequencing architectures or to improve the annotation capability, and the ability to simulate reads for use in model
training. A more detailed overview of Tranquillyzer can be [found in the documentation](docs/index.qmd).

# Quick Start and General Usage

For a guide to getting started with Tranquillyzer, see the [Quick Start guide](docs/webpages/quick_start.qmd). For more
detailed notes on using Tranquillyzer, see the [Usage page](docs/webpages/usage.qmd).

# Installation

Tranquillyzer is available through a variety of methods. See the [Installation](docs/webpages/install.qmd) page for
details.

# Issues

Issues can be opened on GitHub: <https://github.com/huishenlab/tranquillyzer/issues>.

# Acknowledgements

- This work is supported by National Institutes of Health grant UM1DA058219.
