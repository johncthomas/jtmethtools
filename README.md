# Overview
`jtmethtools` is a package to facilitate the analysis of sequence based methylation data. It builds on [Bismark](https://felixkrueger.github.io/Bismark/) outputs and provides scripts for 
generating detailed data tables and plots.

It also provides an API for custom analyses of methylation data, described in detail [here](https://github.com/johncthomas/jtmethtools/deployments/github-pages)

# Installation

There are many options for installing packages. This script creates a fresh virtual environment and installs it there.
```bash
git clone https://github.com/johncthomas/jtmethtools.git
cd jtmethtools
python -m venv venv
source venv/bin/activate
pip install .
```

# Command line tools

These commands are available from the command line after installing `jtmethtools`. 
They are designed for common tasks that don't require custom scripting, or for use in pipelines.

| Command                      | What it does                                                                                                                                                                                     |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `jtm-write-methylation-data` | BAM → parquet methylation dataset                                                                                                                                                                |
| `jtm-read-stats`             | Per-read methylation statistics table                                                                                                                                                            |
| `jtm-rs-pos-beta`            | Methylation rate by read position (end-repair bias detection)                                                                                                                                    |
| `jtm-stats-in-regions`       | Per-region methylation statistics                                                                                                                                                                |
| `jtm-write-mixed-bam`        | Create a BAM by sampling reads from multiple input BAMs                                                                                                                                          |
| `jtm-write-mixed-dataset`    | Create a synthetic `MethylationDataset` from multiple datasets                                                                                                                                   |

Check `jtm-<command> --help` for usage instructions and options for each command. In general, outputs are written
to a given output directory and file name is inhereted from the input file, but can be overwritten with `--sample-name`.
 This is designed for concurrent processing of multiple samples without worrying about file names.

Some scripts allow filtering by regions, but this is significantly slower than using `samtools view` to subset the BAM first, 
so I recommend that. 


## Data tables
`jtm-write-methylation-data` generates two Parquet tables, `locus_data.parquet` and `read_data.parquet`. 
These tables can be read in R using `arrow::read_parquet(pq_filepath)` or as a dataset object in Python using 
`jtmethtools.MethylationDataset.from_dir(pq_directory)`. 

The tables have the following structure:

| **Locus data**                 | Per-CpG (or CpH) observations |
|--------------------------------|---|
| `AlignmentIndex`               | Links to `read_data` |
| `Chrm`, `Position`             | Genomic locus |
| `BismarkCode`                  | Z/z/H/h/X/x/U/u (methylation context and state) |
| `MetCpG`                       | Boolean: is this a methylated CpG? |
| `ReadNucleotide`, `PhredScore` | Base call and quality at this locus |

| **Read data**        | Per-read metadata |
|----------------------|---|
| `AlignmentIndex`     | Unique read identifier |
| `Chrm`, `Start`, `End` | Alignment span |
| `MappingQuality`     | MAPQ |


use `jtm-<command> --help` for usage instructions and options for each command.

----


# Package `jtmethtools` — API Overview

The Python API provides an interface to do custom analyses.

```python
import jtmethtools as jtm
```


## Working with Alignments (`jtmethtools.alignments`)
To avoid loading giant tables you can iterate through a BAM file, one read (or read pair) at a time, 
generating Alignment objects. Data from paired reads are merged together.

## `Alignment`

A wrapper around one or two `pysam.AlignedSegment` objects (single-end or paired-end) that resolves overlapping mate pairs into a single set of per-locus values: 

- **`locus_methylation`** — `{reference_position: bismark_code}` for every cytosine context in the read.
- **`locus_quality`** — `{reference_position: phred_score}`.
- **`locus_nucleotide`** — `{reference_position: base}`.

By default values in the read with the highest PHRED score is used at each position for overlapping mates, 
or optionally you can prefer Read 1.

As an example of possible utility, this snippet would count the number of methylated/unmethylated CpHs as a function of PHRED score:

```python
import numpy as np
import jtmethtools as jtm

ch_count_arrays = {
    "methylated": np.zeros(60, dtype=int),
    "unmethylated": np.zeros(60, dtype=int),
}

for alignment in jtm.iter_bam("sample.bam", paired_end=True):
    for pos, code in alignment.locus_methylation.items():
        if code in "Hh":  # CpH context
            is_methylated = 'methylated' if code.isupper() else 'unmethylated'
            phred = alignment.locus_quality[pos]
            ch_count_arrays[is_methylated][phred] += 1
```

The underlying `pysam` objects are accessible as `alignment.a` and `alignment.a2`, so you can use any PySam functionality you need, to filter by mapping quality, check flags, etc.


## Working with Bismark tables

Convenience functions for working with Bismark-style tables, coverage and methylation calls:
- `read_cov`, `write_cov` & `read_bismark_calls_table` return Pandas DataFrames with appropriate dtypes and column names.
- `sum_strands` sums methylation counts from both posistions in a CpG.

```python
import jtmethtools as jtm
cov = jtm.read_cov("sample.cov.gz")
cpg_index = jtm.CpGIndex.from_fasta("hg38.fa")
cov = jtm.sum_strands(cov, cpg_index.locus2index)
```

## `MethylationDataset`

Data tables described above are part of the class.

### Creating a dataset from a BAM

```python

dataset = jtm.methylation_data.process_bam_methylation_data(
    bamfn="sample.bam",
    paired_end=True,
    min_mapq=20,
)
```

This is memory-efficient internally (chunked Arrow tables, dictionary encoding for chromosomes and Bismark codes).

#### Read/write

```python
dataset.write_to_dir("output/sample_name/")
dataset = jtm.methylation_data.MethylationDataset.from_dir("output/sample_name/")
```

Writes `locus_data.parquet`, `read_data.parquet`, and `metadata.json`. The parquet format preserves categorical types and is fast to reload.


#### Read sampling and synthetic datasets

For benchmarking, simulation, or creating mixed samples with known proportions:

```python
# Sample 5000 reads (with or without replacement)
subset = jtm.methylation_data.sample_reads(dataset, n_reads=5000, with_replacement=True, seed=42)

# Create a synthetic mixture: 70% from sample A, 30% from sample B
synthetic = jtm.methylation_data.synthetic_sample(
    inputs=[
        ("data/sampleA/", 0.7, True),   # (path, proportion, with_replacement)
        ("data/sampleB/", 0.3, True),
    ],
    target_reads=10000,
)
```

#### Coverage file output

```python
jtm.methylation_data.write_coverage(dataset.locus_data, "sample.cov.gz")
```

Aggregates locus-level CpG observations into a Bismark-style coverage file (Chrm, Start, End, Perc, Met, Unmet).


## Core Data Structures (`jtmethtools.classes`)

These are available directly at the package level (e.g. `jtm.Genome`, `jtm.CpGIndex`).

### `Genome`

An immutable wrapper around a dictionary of chromosome → sequence. Load a reference genome and use it to locate CpG dinucleotides:

```python
genome = jtm.Genome.from_fasta("hg38.fa")           
# canonical chromosomes only by default, these are the numbered autosomes, chromosomes, plus mitochondrial.
genome = jtm.Genome.from_fasta("hg38.fa", cannonical_only=False)
cpg_index = genome.get_cpg_index(one_indexed=True)   # build a CpG index
```

Chromosome names are handled flexibly — `harmonise_chrm_names()` creates a new `Genome` where both `"chr1"` and `"1"` map to the same sequence, so you don't have to worry about naming conventions across tools.

### `CpGIndex`

A bidirectional mapping between genomic CpG positions and integer indices. Each CpG dinucleotide gets a unique integer: 
*both* the C and G positions map to the same index.

```python
# Build object from FASTA. It's recommended to pickle this for later use as it can be slow to build.
cpg_index = jtm.CpGIndex.from_fasta("hg38.fa", make_one_indexed=True)
cpg_index = jtm.CpGIndex.from_file("hg38.cpg_index.pickle")

# Look up a CpG by position
idx = cpg_index.locus2index[("chr1", 10469)]

```

