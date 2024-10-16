import typing
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Collection, Tuple, Mapping

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pysam
from loguru import logger
from pysam import AlignedSegment, AlignmentFile, AlignmentHeader
from dataclasses import dataclass, field
import pyarrow as pa

def iter_pe_bam(
        bam:str|AlignmentFile,
        start_stop:Tuple[int, int]=(0, np.inf)
) -> Tuple[AlignedSegment, AlignedSegment|None]:
    """Iterate over a paired-end bam file, yielding pairs of alignments.
    Where a read is unpaired, yield (alignment, None).

    Use start_stop for splitting a bam file for, e.g. multiprocessing.
    """
    #todo test iter_pe_bam
    if type(bam) is not AlignmentFile:
        bam = pysam.AlignmentFile(bam, 'rb')

    if not bam.header.get('HD', {}).get('SO', 'Unknown') == 'queryname':
        raise RuntimeError(f'BAM file must be sorted by queryname')

    aln_prev: pysam.AlignedSegment | None = None
    for i, aln_current in enumerate(bam):
        logger.debug(f'Alignment #{i}')
        if i < start_stop[0]:
            continue
        elif i >= start_stop[1]:
            return None

        elif aln_prev is None:
            aln_prev = aln_current
            continue
        elif aln_current.query_name == aln_prev.query_name:
            yield aln_current, aln_prev
            aln_prev = None
        else:
            yield aln_prev, None
            aln_prev = aln_current

    if aln_prev is not None:
        yield aln_prev, None

@dataclass(slots=True)
class Loci:
    chrm:str
    start:int
    end:int

    def to_kwargs(self):
        """dict with keys chrm, start, end."""
        return dict(chrm=self.chrm, start=self.start, end=self.end)

#
# vectors, read and locus level :
#   readid[int], chrm[str], locus[int], state[bool]
#   index of read start-end loci
# methods:
#   read_by_id: returns vector of loci and state
#   met_in_window: returns vector of loci x reads within window
#   reads_by
# metadata:
#   input file paths
#   bam header
#   hash of input bam
# Vectors etc as parquet. Everything gets put in a zip
ReadID = int
ChrmName = str
Locus = int
MethylationStatus = bool

@dataclass
class MethylationData:
    readid: NDArray[int]
    readid_by_chrm: Mapping[ChrmName, NDArray[int]]
    locus: Mapping[ChrmName, NDArray[Locus]]
    methylation: Mapping[ChrmName, NDArray[MethylationStatus]]
    read_location: Mapping[ReadID, Tuple[ChrmName, Locus, Locus]]

    def iter_reads(self) -> NDArray[bool]:
        for rid in self.readid:
            yield self.methylation[rid]

    def metylation_by_readid(
            self, read_id: int
    ) -> NDArray[bool]:
        chrm, start, stop = self.read_location[read_id]
        return self.methylation[chrm][start:stop]

    @lru_cache(maxsize=128)
    def loci_mask(self, chrm, start, end):
        loci = self.locus[chrm]
        mask = (loci >= start) & (loci < end)
        return mask

    @lru_cache(maxsize=10)
    def read_methylation_at_loci(
            self, chrm:ChrmName, start:Locus, End:Locus
    ) -> tuple[NDArray[int], NDArray[bool], NDArray[int]]:
        """Returned arrays give loci, and methylation state at
        those loci for each read that overlaps the given range."""
        mask = self.loci_mask(chrm, start, End)
        loci = self.locus[chrm][mask]
        met = self.methylation[chrm][mask]
        rids = self.readid_by_chrm[chrm][mask]
        return rids, met, loci

    @lru_cache
    def image(self, ):

        # in:
            # positionsA = np.array([100, 105, 110, 121, 100, 110, 121, 122])
            # stateA = np.array([1, 2, 3, 4, 5, 6, 7, 8])
            # vecid = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        # out:
        #   [[1, 2, 3, 4, nan],
        #    [5, nan, 6, 7, 8]



@dataclass(slots=True)
class Pileup:
    data: MethylationData
    loci: Loci

    #@cached_property # req attrs
    def mask(self):
        return self.data.loci_mask(**self.loci.to_kwargs())

    # @cached_property # req attrs
    def image(self):
        arr = rearrange_data(
            self.data.read_methylation_at_loci(
                **self.loci.to_kwargs()
            )
        )




def rearrange_data(positions, states, rowids):
    """
    Rearranges the given state data into a 2D array aligned by positions
    and organized by rowids.

    Parameters:
    positions (np.array): Array of positions.
    states (np.array): Array of state values.
    rowids (np.array): Array of row identifiers.

    Returns:
    np.array: 2D array with state values aligned by positions and organized by rowids.
    """
    # Unique positions and rows


    unique_positions, position_indices = np.unique(positions, return_inverse=True)
    unique_rowids, row_indices = np.unique(rowids, return_inverse=True)

    # Initialize the output array with NaN values
    output_array = np.full((len(unique_rowids), len(unique_positions)), np.nan)

    # Populate the output array directly using the indices
    output_array[row_indices, position_indices] = states

    return output_array.astype(float)  # Ensure the array is of type float for NaN compatibility

# # Example usage
# positionsA = np.array([100, 105, 110, 121, 100, 110, 121, 122])
# stateA = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# rowid = np.array([1, 1, 1, 1, 2, 2, 2, 2])
#
# result_array = rearrange_data(positionsA, stateA, rowid)

