import typing
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Collection, Tuple, Mapping

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pysam
from loguru import logger
from pyarrow.ipc import ReadStats
from pysam import AlignedSegment, AlignmentFile, AlignmentHeader
#from dataclasses import dataclass, field
import pyarrow as pa
import pyarrow.compute as pcomp
from attrs import field, define

from alignments import *


@define(slots=True)
class Loci:
    start:int
    end:int
    chrm:str

    def to_kwargs(self):
        """dict with keys chrm, start, end."""
        return dict(chrm=self.chrm, start=self.start, end=self.end)

    def to_tuple(self):
        """Returns: (start, end, chrm)"""
        return (self.start, self.end, self.chrm)


_ntcodes = dict(zip('ACGTN', np.array([1, 2, 3, 4, 0], dtype=np.uint8)))
NT_CODES = _ntcodes | {v:k for k, v in _ntcodes.items()}

_bsmk_codes = dict(zip('.ZHXU', np.array([0, 1, 2, 3, 4], dtype=np.uint8)))
BISMARK_CODES = _bsmk_codes | {v:k for k, v in _bsmk_codes.items()}


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


# **NOTE** If one of these changes, also change the numpy ndarrays
# used as intermediates in creation.
ReadIDArray = pa.UInt32Array
LocusArray = pa.UInt32Array
ChrmArray = pa.UInt8Array
MapQArray = pa.UInt8Array
NucleotideArray = pa.UInt8Array
MethylationArray = pa.UInt8Array
PhredArray = pa.UInt8Array



class ReadMetadata:
    def __init__(
            self,
            table:pa.Table
    ):
        """Table holds start, end, chromosome, mapping quality,
        optionally original str read name"""

        self.table = table

    @property
    def readID(self) -> ReadIDArray:
        return self.table.column("readID")

    @property
    def start(self) -> LocusArray:
        return self.table.column("start")

    @property
    def end(self) -> LocusArray:
        return self.table.column("end")

    @property
    def chrm(self) -> ChrmArray:
        return self.table.column("chrm")

    @property
    def mapping_qual(self) -> MapQArray:
        return self.table.column("mapping_quality")

    @property
    def read_name(self) -> pa.StringArray:
        return self.table.column("read_name")

    @lru_cache(255)
    def loci_from_readID(self, readID:int) -> Loci:
        row = self.table.slice(readId, readID)
        return Loci(
            start=row.column('start'),
            end=row.columns('end'),
            chrm=row.columns('chrm')
        )

    @lru_cache(255)
    def readIDs_from_loci(self, start:int, end:int, chrm:int) -> pa.Int32Array:

        condition1 = pc.less_equal(self.table.column("start"), pa.scalar(end))
        condition2 = pc.greater_equal(self.table.column("end"), pa.scalar(start))
        condition3 = pc.equal(self.table.column("chrm"), pa.scalar(chrm))

        # combine the conditions
        combined_condition = pc.and_(pc.and_(condition1, condition2), condition3)

        # Filter the table to get rows where all conditions are true
        filtered_readIDs = pc.filter(self.table.column("readID"), combined_condition)

        # Return the filtered readIDs (which will be a pyarrow.Array)
        return filtered_readIDs


@define
class LocusData:

    table: pa.Table

    @property
    def nucleotide(self) -> NucleotideArray:
        return self.table.column("nucleotide")

    @property
    def phred_scores(self) -> PhredArray:
        return self.table.column("phred_scores")

    @property
    def chrm(self) -> ChrmArray:
        return self.table.column("chrm")

    @property
    def locus(self) -> LocusArray:
        return self.table.column("locus")

    @property
    def readID(self) -> ReadIDArray:
        return self.table.column("readID")

    @property
    def methylation(self) -> MethylationArray:
        return self.table.column("methylation")

    @property
    def is_insertion(self) -> pa.BooleanArray:
        return self.table.column('is_insertion')


@define
class ReadData:
    def __init__(self, loci:LocusData, read:ReadMetadata):
        self.loci = locus_table
        self.read = read


    def loci_by_readID(self, readID:int) -> Loci:
        return self.read.loci_from_readID(readID)

    @lru_cache(16)
    def reads_at_loci(self, loci:Loci):
        self.read.reads_at_loci(loci)

    @lru_cache(16)
    def readIDs_from_loci(self,):
        self.read.reads_at_loci()

    # @lru_cache(maxsize=128)
    # def loci_mask(self, chrm, start, end):
    #     loci = self.locus[chrm]
    #     mask = (loci >= start) & (loci < end)
    #     return mask

    # def iter_reads(self) -> NDArray[bool]:
    #     for rid in self.readid:
    #         yield self.methylation[rid]


    # def metylation_by_readid(
    #         self, read_id: int
    # ) -> NDArray[bool]:
    #     chrm, start, stop = self.read_location[read_id]
    #     return self.methylation[chrm][start:stop]



    # @lru_cache(maxsize=10)
    # def read_methylation_at_loci(
    #         self, chrm:ChrmName, start:Locus, End:Locus
    # ) -> tuple[NDArray[int], NDArray[bool], NDArray[int]]:
    #     """Returned arrays give loci, and methylation state at
    #     those loci for each read that overlaps the given range."""
    #     mask = self.loci_mask(chrm, start, End)
    #     loci = self.locus[chrm][mask]
    #     met = self.methylation[chrm][mask]
    #     rids = self.readid_by_chrm[chrm][mask]
    #     return rids, met, loci

def _encode_string(string:str, codes:dict) -> NDArray[np.uint8]:
    """Perform substitution of string to values in codes."""
    nt_arr = np.zeros((len(string),), dtype=np.uint8)
    for i, n in enumerate(string):
        nt_arr[i] = codes[n]
    return nt_arr

def encode_nt_str(nts:str) -> NDArray[np.uint8]:
    return _encode_string(nts, NT_CODES)

def encode_metstr_bismark(metstr:str) -> NDArray[np.uint8]:
    return _encode_string(nts, BISMARK_CODES)


def print_memory_footprint():
    import psutil
    import os

    # Get the current process
    process = psutil.Process(os.getpid())

    # Get memory usage in bytes (can also use process.memory_info().rss for just resident memory)
    memory_usage = process.memory_info().rss  # in bytes

    # Convert to megabytes (MB) for easier readability
    memory_usage_mb = memory_usage / (1024 ** 2)

    print(f"Memory usage: {memory_usage_mb:.2f} MB")

def process_bam_to_ndarrays(bamfn, regionsfn:str) -> dict[str, NDArray]:
    # for recording things that should go in the final object
    data = {}

    # get the number of reads and the number of aligned bases
    n_reads = 0
    n_bases = 0
    paired = False
    regions = Regions.from_file(regionsfn)

    chrm_map = {}
    for i, c in enumerate(Regions.chromsomes):
        chrm_map[i] = c
        chrm_map[c] = i

    data['chrm_map'] = chrm_map

    print('Before creating creating empty arrays:')
    print_memory_footprint()

    for aln in iter_bam(bamfn, paired_end=paired):
        aln:Alignment

        hit_regns = aln.hit_regions(regions)

        if hit_regns:
            n_reads += 1
            n_bases += aln.a.query_length


    read_arrays = dict(
        readID=np.zeros(dtype=np.uint32, shape=(n_reads,)),
        start=np.zeros(dtype=np.uint32, shape=(n_reads,)),
        end=np.zeros(dtype=np.uint32, shape=(n_reads,)),
        chrm=np.zeros(dtype=np.uint8, shape=(n_reads,)),
        mapping_qual=np.zeros(dtype=np.uint8, shape=(n_reads,)),
    )
    loci_arrays = dict(
        nucleotide=np.zeros(dtype=np.uint8, shape=(n_bases,)),
        phred_scores=np.zeros(dtype=np.uint8, shape=(n_bases,)),
        chromosome=np.zeros(dtype=np.uint8, shape=(n_bases,)),
        locus=np.zeros(dtype=np.uint32, shape=(n_bases,)),
        readID=np.zeros(dtype=np.uint32, shape=(n_bases,)),
        methylation=np.zeros(dtype=np.uint8, shape=(n_bases,)),
        is_insertion=np.zeros(dtype=np.uint8, shape=(n_bases,))
    )

    print('After:')
    print_memory_footprint()

    read_i = -1
    locus_i_next = 0
    for readID, aln in enumerate(iter_bam(bamfn, paired_end=paired)):
        aln:Alignment

        if not aln.hit_regions(regions):
            continue

        # deal with the table indicies
        read_i += 1
        locus_i = locus_i_next
        locus_j = locus_i + aln.a.query_length
        locus_i_next = locus_j

        read_arrays['readID'][read_i] = readID
        read_arrays['mapping_qual'][read_i] = aln.a.mapping_quality
        read_arrays['start'][read_i] = aln.a.reference_start
        read_arrays['end'][read_i] = aln.a.reference_end
        read_arrays['chrm'][read_i] = aln.a.reference_id
        #read_arrays['read_name'][read_i] = a.a.query_name

        loci_arrays['nucleotide'][locus_i:locus_j] = encode_nt_str(aln.a.query_sequence, )
        loci_arrays['phred_scores'][locus_i:locus_j] = np.array(aln.a.query_qualities, dtype=np.uint8)
        loci_arrays['chromosome'][locus_i:locus_j] = aln.a.reference_id
        loci_arrays['locus'][locus_i:locus_j] = aln.a.get_reference_positions()
        loci_arrays['readID'][locus_i:locus_j] = readID
        loci_arrays['methylation'][locus_i:locus_j] = encode_metstr_bismark(aln.metstr)
        loci_arrays['is_insertion'][locus_i:locus_j] = np.array(get_insertion_mask(aln.a), dtype=np.uint8)

    pa_read_arrays = {}
    # create pa arrays, using correct dtype
    pa_read_arrays['readID'] = ReadIDArray(read_arrays['readID'])
    pa_read_arrays['start'] = LocusArray(read_arrays['start'])
    pa_read_arrays['end'] = LocusArray(read_arrays['end'])
    pa_read_arrays['chrm'] = ChrmArray(read_arrays['chrm'])
    pa_read_arrays['mapping_qual'] = MapQArray(read_arrays['mapping_qual'])

    # get things out of memory asap
    del read_arrays
    read_metadata = ReadMetadata(
        pa.Table.from_pydict(pa_read_arrays)
    )
    del pa_read_arrays

    pa_loci_arrays = {}
    pa_loci_arrays['nucleotide'] = NucleotideArray(loci_arrays['nucleotide'])
    pa_loci_arrays['phred_scores'] = PhredArray(loci_arrays['phred_scores'])
    pa_loci_arrays['chromosome'] = ChrmArray(loci_arrays['chromosome'])
    pa_loci_arrays['locus'] = LocusArray(loci_arrays['locus'])
    pa_loci_arrays['readID'] = ReadIDArray(loci_arrays['readID'])
    pa_loci_arrays['methylation'] = MethylationArray(loci_arrays['methylation'])
    pa_loci_arrays['is_insertion'] = MethylationArray(loci_arrays['is_insertion'])
    del loci_arrays

    read_metadata = ReadMetadata(
        pa.Table.from_pydict(pa_read_arrays)
    )
    loci_data = LocusData(
        pa.Table.from_pydict(pa_loci_arrays)
    )

    return dict(read_metadata=read_metadata, loci_data=loci_data)



# THINGS to TEST
# do the is_insertion == 1 match the insertions in the locus.










@dataclass(slots=True)
class Pileup:
    data: LocusData
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

process_bam_to_ndarrays(
    '/home/jaytee/DevLab/NIMBUS/Data/test/bismark_10k.sorted.bam',
    '/home/jaytee/DevLab/NIMBUS/Reference/regions-table.canary.4k.tsv'
)