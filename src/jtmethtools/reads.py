import pathlib
from os import PathLike
import os
from pathlib import Path
import typing
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Collection, Tuple, Mapping
import tarfile

import attrs
from loguru import logger

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pysam
from pysam import AlignedSegment, AlignmentFile, AlignmentHeader

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import parquet

from attrs import field, define

from jtmethtools.alignments import (
    Alignment,
    iter_bam,
    Regions,
)

Pathesque = str | Path | PathLike[str]

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


@lru_cache(maxsize=128)
def loci_mask(
        target_loci:LocusArray, target_chrm:ChrmArray,
        start:int, end:int, chrm:int):

    # (loci >= start) & (loci < end)
    loc_m = pc.and_(pc.greater_equal(target_loci, pa.scalar(start)),
                    pc.less(target_loci, pa.scalar(end)))
    chrm_m = pc.equal(target_chrm, pa.scalar(chrm))
    return pc.and_(loc_m, chrm_m)




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
        row = self.table.slice(readID, readID)
        return Loci(
            start=row.column('start'),
            end=row.columns('end'),
            chrm=row.columns('chrm')
        )

    @lru_cache(255)
    def read_ids_at_loci(self, start:int, end:int, chrm:int) -> pa.Int32Array:
        condition1 = pc.less_equal(self.table.column("start"), pa.scalar(end))
        condition2 = pc.greater_equal(self.table.column("end"), pa.scalar(start))
        condition3 = pc.equal(self.table.column("chrm"), pa.scalar(chrm))

        # combine the conditions
        combined_condition = pc.and_(pc.and_(condition1, condition2), condition3)

        # Filter the table to get rows where all conditions are true
        filtered_readIDs = pc.filter(self.table.column("readID"), combined_condition)

        # Return the filtered readIDs (which will be a pyarrow.Array)
        return filtered_readIDs

    def to_parquet(self, fn: str|Path):
        """Write table in Parquet format"""
        parquet.write_table(self.table, fn)

    @classmethod
    def from_parquet(cls, fn: Pathesque):
        """Read table from Parquet format"""
        return cls(parquet.read_table(fn))

    def to_dir(self, directory: str | Path):
        directory = pathlib.Path(directory)
        os.makedirs(directory, exist_ok=True)
        self.to_parquet(directory / 'read-metadata.parquet')
        # with open(directory / 'read-attributes.dict', 'w') as f:
        #     f.write(str({...}))

    @classmethod
    def from_dir(cls, directory: str | Path):
        directory = pathlib.Path(directory)
        table = parquet.read_table(directory / 'loci-data.parquet')
        # with open(directory / 'read-attributes.dict'', 'r') as f:
        #     d = eval(f.read())
        return cls(table)


@define
class LocusData:

    table: pa.Table
    chrm_map: dict

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

    def to_parquet(self, fn:Path|str):
        """Write table in Parquet format"""
        parquet.write_table(self.table, fn)

    def to_dir(self, directory:str|Path):
        directory = pathlib.Path(directory)
        os.makedirs(directory, exist_ok=True)
        self.to_parquet(directory/'loci-data.parquet')
        with open(directory/'locus-attributes.dict', 'w') as f:
            f.write(str({'chrm_map':self.chrm_map}))

    @classmethod
    def from_dir(cls, directory:str|Path):
        directory = pathlib.Path(directory)
        table = parquet.read_table(directory/'loci-data.parquet')
        with open(directory/'locus-attributes.dict', 'r') as f:
            d = eval(f.read())
        return cls(table, **d)

    @classmethod
    def from_parquet(cls, fn:str|Path):
        """Read table from Parquet format"""
        return cls(parquet.read_table(fn))



@define
class ReadData:
    loci:LocusData
    read:ReadMetadata
    bam_fn:str
    bam_header:str
    version:int = 1 #increment when attributes change

    def _get_nontable_attr(self):
        d = attrs.asdict(self)
        del d['loci']
        del d['read']
        return d

    def loci_by_readID(self, readID:int) -> Loci:
        return self.read.loci_from_readID(readID)

    @lru_cache(16)
    def reads_at_loci(self, start, end, chrm):
        reads_ids = self.read.read_ids_at_loci(start, end, chrm)

    def to_dir(self, directory:Pathesque):
        directory = pathlib.Path(directory)
        os.makedirs(directory, exist_ok=True)
        self.loci.to_dir(directory)
        self.read.to_dir(directory)
        # add the rest
        d = self._get_nontable_attr()

        with open(directory/'short-fields.dict', 'w') as f:
            f.write(str(d))

    @classmethod
    def from_dir(cls, directory:Pathesque):
        directory = pathlib.Path(directory)
        loci = LocusData.from_dir(directory)
        read = ReadMetadata.from_dir(directory)

        with open(directory/'short-fields.dict', 'r') as f:
            d = eval(f.read())

        return cls(loci, read, **d)

    def print_heads(self, n=5) -> None:
        print('Read table:')
        print(self.read.table.slice(0, n))
        print('\n++++\n')
        print('Loci table:')
        print(self.loci.table.slice(0, n))


    def loci_mask(self, start:int, end:int, chrm:str):
        return loci_mask(self.loci.locus, self.loci.chrm,
                  start, end, self.loci.chrm_map[chrm])




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
        try:
            nt_arr[i] = codes[n]
        except KeyError:
            pass
    return nt_arr

def encode_nt_str(nts:str) -> NDArray[np.uint8]:
    return _encode_string(nts, NT_CODES)

def encode_metstr_bismark(metstr:str) -> NDArray[np.uint8]:
    return _encode_string(metstr, BISMARK_CODES)


def get_insertion_mask(a:AlignedSegment):
    """
    Returns a list where:
    - 0 indicates a matched base (CIGAR operation 0),
    - 1 indicates an inserted base (CIGAR operation 1).

    Args:
        a (pysam.AlignedSegment): The aligned read.

    Returns:
        list: A list of 0s and 1s indicating matched and inserted bases.
    """
    match_insertion_list = []

    for (operation, length) in a.cigartuples:
        if operation == 0:  # Match
            match_insertion_list.extend([0] * length)
        elif operation == 1:  # Insertion
            match_insertion_list.extend([1] * length)
        if not len(match_insertion_list) == a.query_length:
            s = f"Query length did not match calculation. cigar={a.cigartuples}, readName={a.query_name}"
            raise ValueError(s)
        # else: continue or handle other operations based on requirements

    return match_insertion_list


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

def process_bam_to_readdata(bamfn, regionsfn:str) -> ReadData:

    # get the number of reads and the number of aligned bases
    n_reads = 0
    n_bases = 0
    paired = False
    if regionsfn.endswith('.tsv'):
        regions = Regions.from_file(regionsfn)
    else:
        regions = Regions.from_bed(regionsfn)

    chrm_map = {}
    for i, c in enumerate(regions.chromsomes):
        chrm_map[i] = c
        chrm_map[c] = i



    print('Before creating creating empty arrays:')
    print_memory_footprint()

    for aln in iter_bam(bamfn, paired_end=paired):
        aln:Alignment

        hit_regns = aln.hit_regions(regions)

        if hit_regns:
            n_reads += 1
            n_bases += aln.a.query_length

    # this dtype has implications...
    locus_dtype = np.uint32

    read_arrays = dict(
        readID=np.zeros(dtype=np.uint32, shape=(n_reads,)),
        start=np.zeros(dtype=locus_dtype, shape=(n_reads,)),
        end=np.zeros(dtype=locus_dtype, shape=(n_reads,)),
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


    max32 = np.iinfo(locus_dtype).max
    read_i = -1
    locus_i_next = 0
    alignmentfile = pysam.AlignmentFile(bamfn)
    for readID, aln in enumerate(iter_bam(alignmentfile, paired_end=paired)):
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
        ref_p = aln.a.get_reference_positions(full_length=True)
        ref_p = np.array([locus_dtype(n) if n is not None else max32 for n in ref_p])
        loci_arrays['locus'][locus_i:locus_j] = ref_p
        loci_arrays['readID'][locus_i:locus_j] = readID
        loci_arrays['methylation'][locus_i:locus_j] = encode_metstr_bismark(aln.metstr)
        loci_arrays['is_insertion'][locus_i:locus_j] = ref_p ==  max32

    pa_read_arrays = {
        'readID': pa.array(read_arrays['readID']),
        'start': pa.array(read_arrays['start']),
        'end': pa.array(read_arrays['end']),
        'chrm': pa.array(read_arrays['chrm']),
        'mapping_qual': pa.array(read_arrays['mapping_qual'])
    }
    # get things out of memory asap
    del read_arrays

    read_metadata = ReadMetadata(
        pa.Table.from_pydict(pa_read_arrays)
    )
    del pa_read_arrays

    pa_loci_arrays = {
        'nucleotide': pa.array(loci_arrays['nucleotide']),
        'phred_scores': pa.array(loci_arrays['phred_scores']),
        'chromosome': pa.array(loci_arrays['chromosome']),
        'readID': pa.array(loci_arrays['readID']),
        'methylation': pa.array(loci_arrays['methylation']),
        'is_insertion': pa.array(loci_arrays['is_insertion'])
    }
    print('max point probably')
    print_memory_footprint()
    del loci_arrays

    loci_data = LocusData(
        pa.Table.from_pydict(pa_loci_arrays),
        chrm_map=chrm_map,
    )
    print('Just before returning')
    print_memory_footprint()

    print('Reads: ', read_i, 'Locus: ', locus_i_next)



    return ReadData(loci_data, read_metadata, bam_fn=bamfn,
                    bam_header=str(alignmentfile.header))


# @define(slots=True)
# class Pileup:
#     data: LocusData
#     loci: Loci
#
#     #@cached_property # req attrs
#     def mask(self):
#         return self.data.loci_mask(**self.loci.to_kwargs())
#
#     # @cached_property # req attrs
#     def image(self):
#         arr = rearrange_data(
#             self.data.read_methylation_at_loci(
#                 **self.loci.to_kwargs()
#             )
#         )






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

def ttests(test_bam_fn, test_regions_fn, testoutdir,
           *, delete_first=False):
    import datetime
    if delete_first:
        os.rmdir(testoutdir, )
    # get current time for timedelta
    start = datetime.datetime.now()
    rd = process_bam_to_readdata(
        test_bam_fn,
        test_regions_fn
    )
    # time checkpoint
    next1 = datetime.datetime.now()
    print('Time to process:', next1 - start)

    rd.to_dir(testoutdir)
    next2 = datetime.datetime.now()
    print('Time to write:', next2 - next1)

    rd2 = ReadData.from_dir(testoutdir)
    next3 = datetime.datetime.now()
    print('Time to read:' , next3 - next2)
    print('\n')
    rd.print_heads()
    rd2.print_heads()




if __name__ == '__main__':

    bm = '/home/jcthomas/DevLab/NIMBUS/Data/test/bismark_10k.bam'
    #bm = '/home/jcthomas/data/canary/sorted_qname/CMDL19003173_1_val_1_bismark_bt2_pe.deduplicated.bam'
    rg = '/home/jcthomas/DevLab/NIMBUS/Reference/regions-table.canary.4k.tsv'
    out = '/home/jcthomas/DevLab/NIMBUS/Data/test/readdata_structure_test'
    ttests(bm, rg, out)