# `jtmethtools` — API Overview

`jtmethtools` is a Python library for working with bisulfite-sequencing methylation data. It provides data structures and tools that sit between raw Bismark BAM output and downstream analysis, handling the tedious mechanics of parsing alignments, indexing CpG sites, and reshaping data into analysis-ready tables.

```python
import jtmethtools as jtm
```

---

## Core Data Structures (`jtmethtools.classes`)

These are available directly at the package level (e.g. `jtm.Genome`, `jtm.CpGIndex`).

### `Genome`

An immutable wrapper around a dictionary of chromosome → sequence. Load a reference genome and use it to locate CpG dinucleotides:

```python
genome = jtm.Genome.from_fasta("hg38.fa")           # canonical chromosomes only by default
genome = jtm.Genome.from_fasta("hg38.fa", cannonical_only=False)
cpg_index = genome.get_cpg_index(one_indexed=True)   # build a CpG index
```

Chromosome names are handled flexibly — `harmonise_chrm_names()` creates a new `Genome` where both `"chr1"` and `"1"` map to the same sequence, so you don't have to worry about naming conventions across tools.

### `CpGIndex`

A bidirectional mapping between genomic CpG positions and integer indices. Each CpG dinucleotide gets a unique integer, and *both* the C and G positions map to the same index — so lookups work regardless of which strand a read is on.

```python
# Build from genome, from FASTA, or load a pre-built pickle (much faster for large genomes)
cpg_index = jtm.CpGIndex.from_fasta("hg38.fa", make_one_indexed=True)
cpg_index = jtm.CpGIndex.from_file("hg38.cpg_index.pickle")

# Look up a CpG by position
idx = cpg_index.locus2index[("chr1", 10469)]

# Restrict to CpGs within target regions
filtered = cpg_index.filter_cpg_index_by_regions(regions)
```

Pickling and loading (`to_file` / `from_file`) is the recommended approach for large genomes — building from FASTA is slow; loading from pickle is near-instant.

### `Regions`

Genomic intervals (e.g. target panels, CpG islands, promoters). Constructed from BED files or DataFrames, and used throughout the library for filtering:

```python
regions = jtm.Regions.from_file("targets.bed")
regions = jtm.Regions.from_df(df)  # expects columns: Chrm, Start, End, Name

# Point query: which region (if any) overlaps a locus?
region_name = regions.region_at_locus("chr1", 10469)
```

Chromosome naming is handled transparently — `"chr1"` and `"1"` both work regardless of how the BED was formatted.

### `LociRange`

A lightweight container for a single genomic interval (`chrm`, `start`, `end`, optional `name`). Mostly used internally when iterating over `Regions`.

---

## Working with Alignments (`jtmethtools.alignments`)

### `Alignment`

A wrapper around one or two `pysam.AlignedSegment` objects (single-end or paired-end) that resolves overlapping mate pairs into a single set of per-locus values:

- **`locus_methylation`** — `{reference_position: bismark_code}` for every cytosine context in the read.
- **`locus_quality`** — `{reference_position: phred_score}`, with optional quality recalculation for overlapping paired-end regions using NGmerge-derived profiles.
- **`locus_nucleotide`** — `{reference_position: base}`.

Where mates overlap, the base with the higher PHRED score wins. This is a key problem that `jtmethtools` handles for you — naively counting both mates double-counts loci in the overlap.

### Iteration helpers

```python
# Iterate yielding pysam segment tuples (low-level)
for segments in jtm.iter_bam_segments("sample.bam", paired_end=True):
    a1, a2 = segments  # a2 is None for SE

# Iterate yielding Alignment objects (higher-level, resolves overlaps)
for aln in jtm.iter_bam("sample.bam", paired_end=True):
    met_string = aln.metstr
```

### Utility functions

- **`get_bismark_met_str(segment)`** — extract the Bismark XM tag from a `pysam.AlignedSegment`.
- **`write_bam_from_pysam(out_fn, alignments, header=...)`** — write a collection of `AlignedSegment` objects to a BAM or SAM.
- **`alignment_overlaps_region(segment, regions)`** — check (and return the name of) any region an alignment overlaps.

---

## Methylation Datasets (`jtmethtools.methylation_data`)

This is where most analysis workflows start. The idea is to convert a BAM into a pair of tidy tables — one for per-locus observations and one for per-read metadata — and then work with those tables in pandas.

### `MethylationDataset`

A container holding two DataFrames linked by `AlignmentIndex`:

| **`locus_data`** | Per-CpG (or CpH) observations |
|---|---|
| `AlignmentIndex` | Links to `read_data` |
| `Chrm`, `Position` | Genomic locus |
| `BismarkCode` | Z/z/H/h/X/x/U/u (methylation context and state) |
| `MetCpG` | Boolean: is this a methylated CpG? |
| `ReadNucleotide`, `PhredScore` | Base call and quality at this locus |

| **`read_data`** | Per-read metadata |
|---|---|
| `AlignmentIndex` | Unique read identifier |
| `Chrm`, `Start`, `End` | Alignment span |
| `MappingQuality` | MAPQ |

#### Creating a dataset from a BAM

```python
dataset = jtm.methylation_data.process_bam_methylation_data(
    bamfn="sample.bam",
    paired_end=True,
    regions=regions,               # optional: only include reads overlapping these regions
    min_mapq=20,
    drop_methylated_ch_reads=True, # discard reads with any mCH
    include_unmethylated_ch=False,  # only record CpG and methylated CpH loci
)
```

This is memory-efficient internally (chunked Arrow tables, dictionary encoding for chromosomes and Bismark codes), but returns ordinary pandas DataFrames.

#### Persistence

```python
dataset.write_to_dir("output/sample_name/")
dataset = jtm.methylation_data.MethylationDataset.from_dir("output/sample_name/")
```

Writes `locus_data.parquet`, `read_data.parquet`, and `metadata.json`. The parquet format preserves categorical types and is fast to reload.

#### Filtering

```python
clean = dataset.drop_methylated_CH()  # returns a new dataset without mCH reads
```

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

`AlignmentIndex` values are remapped so the combined dataset is self-consistent.

#### Coverage file output

```python
jtm.methylation_data.write_coverage(dataset.locus_data, "sample.cov.gz")
```

Aggregates locus-level CpG observations into a Bismark-style coverage file (Chrm, Start, End, Perc, Met, Unmet).

---

## CLI Tools

The package installs several command-line entry points for pipeline use:

| Command | What it does |
|---------|-------------|
| `jtm-write-methylation-data` | BAM → parquet methylation dataset |
| `jtm-filter-ch` | Write a new BAM with mCH reads removed |
| `jtm-read-stats` | Per-read methylation statistics table |
| `jtm-rs-pos-beta` | Methylation rate by read position (end-repair bias detection) |
| `jtm-stats-in-regions` | Per-region methylation statistics |
| `jtm-write-mixed-bam` | Create a BAM by sampling reads from multiple input BAMs |
| `jtm-write-mixed-dataset` | Create a synthetic `MethylationDataset` from multiple datasets |
| `jtm-beta-balance` | Resample methylated counts across coverage files so all samples share the same global beta |

---

## Typical Workflow

1. **`jtm-write-methylation-data`** to convert Bismark BAMs into parquet datasets.
2. Load with `MethylationDataset.from_dir()` and do your analysis in pandas — per-read beta, per-region concordance, locus-level filtering by PHRED, etc.
3. Use `CpGIndex` and `Regions` to map between genomic coordinates and structured indices.
4. Use `sample_reads` / `synthetic_sample` / `jtm-beta-balance` for simulation and benchmarking work.
5. Use `write_coverage` or `jtm-beta-balance` to produce coverage files for tools that expect them.

The library is designed so that the heavy I/O (BAM parsing) happens once, and everything downstream operates on efficient tabular data.

