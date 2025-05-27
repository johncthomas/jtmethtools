Methylation data tools. Primarily classes and functions to be built upon by other Python tools. Arrow based tables for efficiently storing and processing Bismark BAMs. Module for producing 
pile-up images of regions for CNNs.

# Classes
Explore alignments.Alignment and classes.*

# Convert a BAM to a parquet tables
Script: `jtm-write-alignment-data`

Outputs two tables, one with locus level (nucleotide, individual CpG, etc.) information, and one with read level 
information, plus metadata.

To load the data in R:
```R
library(arrow)
library(jsonlite)

locusTable <- read_parquet("dataset/locus-table.parquet")
readTable <- read_parquet("dataset/read-table.parquet")
metadata <- fromJSON("dataset/metadata.json")
chrm_ids <- metadata['locus']['chrm_map']
```

Most text data is encoded into integers. These mappings are recorded in the metadata.


# Images for CNN
2D pileups, as binary arrays with values between 0 & 1 representing different sequence features such as methylation state, 
mapping quality and nucleotide sequence.

Paired-end BAMs should be sorted by query name (preferably) or coordinate (may take more memory). Unsorted BAMs
will probably work but use a lot of memory.

## Generation
After installation use `jtm-generate-images run --help` for arguments. Available layers (that can then be passed
to the `--layer` option) can be printed using `jtm-generate-images layers`. `run` produces gzipped TAR files that
contain the binary array and a metadata JSON file, specifying the shape of the array and other things.

`jtm-generate-images` invokes the script `generate_images.py`. 

## Arrays
```python
import jtmethtools as jtm
fn = 'image.region_name.layer.tar.gz'
array, metadata = jtm.images.read_array(fn)

import matplotlib.pyplot as plt
plt.imshow(array, interpolation='nearest', cmap='gray')
```
