r"""
# `jtmethtools`

## Methylation data processing tools for Python

`jtmethtools` is a Python library for working with bisulfite-sequencing methylation data. It builds on PySam to provide data structures and tools that sit between raw Bismark BAM output and downstream analysis, to make it easier to work with methylation data in Python.

It includes:
- `jtmethtools.methylation_data`: Arrow backed DataFrames of methylation data that includes more information, more
efficiently than the standard Bismark tables. Includes some methods for standard opertions like summing strands or filtering by coverage.
- `jtmethtools.alignments`: A wrapper around PySam's `AlignedSegment` objects that resolves overlapping mate pairs into a single set of per-locus values, and provides an iterator for BAM files that generates these objects one read (or read pair) at a time.
 `alignments` is imported into the main `jtmethtools` namespace.
- `jtmethtools.CpGIndex`: A class for indexing CpG positions in a reference genome, to make it easier to sum strands and filter by CpG position.
- `jtmethtools.utils`: A grab bag of utility functions for working with methylation data, including functions for reading/writing Bismark tables, and some helper functions for working with the `Alignment` objects.
- `jtmethtools.pearl_images`: A small collection of functions for visualizing methylation data, including plotting the methylation status of individual reads.
"""


from jtmethtools import (
    alignments,
    methylation_data,
    util,
    pearl_images,

)
from jtmethtools.util import *
from jtmethtools.alignments import *
from jtmethtools.classes import *