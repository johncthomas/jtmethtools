import pathlib
from os import PathLike
import os
from pathlib import Path
import typing
from typing import Collection, Tuple, Mapping, Self
from functools import cached_property, lru_cache


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



# pyarrow bools aren't recognised as bools by python, they
# are Truthy, even when false. So using x == TruePA
TruePA = pa.scalar(True, type=pa.bool_())
FalsePA = pa.scalar(False, type=pa.bool_())

@define(slots=True)
class LociRange:
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

# **NOTE** If one of these changes, also change the numpy ndarrays
# used as intermediates in creation.
ReadIDArray = pa.UInt32Array
LocusArray = pa.UInt32Array
ChrmArray = pa.UInt8Array
MapQArray = pa.UInt8Array
NucleotideArray = pa.UInt8Array
MethylationArray = pa.UInt8Array
PhredArray = pa.UInt8Array


def loci_mask(
        target_loci:LocusArray, target_chrm:ChrmArray,
        start:int, end:int, chrm:int) -> pa.BooleanArray:

    # (loci >= start) & (loci < end)
    loc_m = pc.and_(pc.greater_equal(target_loci, pa.scalar(start)),
                    pc.less(target_loci, pa.scalar(end)))
    chrm_m = pc.equal(target_chrm, pa.scalar(chrm))
    return pc.and_(loc_m, chrm_m)


class ReadMetadata:
    def __init__(
            self,
            table:pa.Table,
            max_mapq=42,
    ):
        """Table holds read level information, columns are
        accessible as properties of this object."""

        self.table = table
        self.max_mapq = max_mapq
        if pc.greater(
                pc.max(self.mapping_quality),
                pa.scalar(max_mapq, type=pa.uint8())
        ) == TruePA:
            max_found = pc.max(self.mapping_quality).as_py()
            raise RuntimeError(
                f"Max MAPQ exceeded ({max_found} > {max_mapq}) , the alignment is using a different"
                " scale and image gen would need to be rewritten"
            )

    def get_col(self, col:str):
        return self.table.column(col)

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
    def mapping_quality(self) -> MapQArray:
        return self.table.column("mapping_quality")

    @property
    def read_name(self) -> pa.StringArray:
        return self.table.column("read_name")

    @property
    def is_forward(self) -> pa.BooleanArray:
        return self.get_col('is_forward')

    @lru_cache(255)
    def loci_from_readID(self, readID:int) -> LociRange:
        row = self.table.slice(readID, readID)
        return LociRange(
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
        table = parquet.read_table(directory / 'read-metadata.parquet')
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
    def position(self) -> LocusArray:
        return self.table.column("position")

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

    def loci_mask(self, loci:LociRange) -> pa.BooleanArray:
        return loci_mask(self.position, self.chrm,
                  loci.start, loci.end, self.chrm_map[loci.chrm])

    @cached_property
    def max_phred(self) -> int:
        return pc.max(self.phred_scores).as_py()

    def filter(self, *args, **kwargs) -> Self:
        """Args passed to self.table.filter(), returns LocusData with
        filtered self.table"""
        return LocusData(self.table.filter(*args, **kwargs), chrm_map=self.chrm_map)

    def remove_insertions(self) -> Self:
        return self.filter(pc.invert(self.is_insertion))

    def window(self, loci:LociRange) -> Self:
        m = self.loci_mask(loci)
        window = self.filter(m)
        return window


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
    locus_data:LocusData
    read_data:ReadMetadata
    bam_fn:str
    bam_header:str
    version:int = 1 #increment when attributes change

    def _get_nontable_attr(self):
        d = attrs.asdict(self)
        del d['locus_data']
        del d['read_data']
        return d

    def loci_by_readID(self, readID:int) -> LociRange:
        return self.read_data.loci_from_readID(readID)

    # @lru_cache(16)
    # def reads_at_loci(self, start, end, chrm):
    #     reads_ids = self.read_data.read_ids_at_loci(start, end, chrm)

    def to_dir(self, directory:Pathesque):
        directory = pathlib.Path(directory)
        os.makedirs(directory, exist_ok=True)
        self.locus_data.to_dir(directory)
        self.read_data.to_dir(directory)
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

    @staticmethod
    def from_bam(
            bamfn:Pathesque,
            regionsfn:Pathesque|None=None
    ):
        self = process_bam_to_readdata(bamfn, regionsfn)
        return self

    def print_heads(self, n=5) -> None:
        print('Read table:')
        print(self.read_data.table.slice(0, n))
        print('\n++++\n')
        print('Loci table:')
        print(self.locus_data.table.slice(0, n))


    def loci_mask(self, loci:LociRange) -> pa.BooleanArray:
        return self.locus_data.loci_mask(loci)

    def window(self, start:int, end:int, chrm:str,) -> Self:
        loci = LociRange(
            start=start,
            end=end,
            chrm=chrm
        )
        mask = self.loci_mask(loci)
        locuswindow = self.locus_data.filter(mask)
        # note: at the moment can't filter read_data cus we rely on the row
        # positions, but it also should be relatively small.
        return ReadData(locuswindow, self.read_data, **self._get_nontable_attr())




class Image:
    __slots__ = ['window', 'positions', 'unique_positions',
                 'unique_rowids', 'position_indices', 'row_indices',
                 '_readid_image', 'width']

    null_grey = 0.2 # value where a base exists but is negative
    strand_colours = (0.8, 1)


    def __init__(self, window:ReadData):
        self.window:ReadData = window

        pos = window.locus_data.position.to_numpy()
        rids =  window.locus_data.readID.to_numpy()

        # add padding, and ensure gaps are shown by creating a
        #  blank read that hits every position between the start
        #  and end of the window.
        start, end = min(pos), max(pos)
        self.width = width = end-start

        padrid = np.max(rids) + 1
        rids = np.concatenate([
            rids,
            np.ones(shape=(width,), dtype=rids.dtype) * padrid,
        ])

        pos = np.concatenate([
            pos, np.arange(start, end, dtype=pos.dtype)
        ])

        # These values used by .protoimage()
        unique_positions, position_indices = np.unique(pos, return_inverse=True)
        unique_rowids, row_indices = np.unique(rids, return_inverse=True)

        self.unique_positions:NDArray = unique_positions
        self.unique_rowids:NDArray = unique_rowids
        self.position_indices:NDArray = position_indices
        self.row_indices:NDArray = row_indices

        # image with readID as values, base for annotating read
        #  level values. Copys provided by method
        self._readid_image = self._protoimage(window.locus_data.readID)

    @staticmethod
    def finish_image(image, fillval=0) -> None:
        """Fill missing (np.nan) positions in image with 0,
        or fillval & remove bottom row (which is the """
        image[np.isnan(image)] = fillval
        return image[:-1]

    @property
    def read_id_image_copy(self):
        return np.copy(self._readid_image)

    def _protoimage(self, states:pa.Array):
        """Create image (array) where values are given by state,
        and missing positions are np.nan
        """
        states = states.to_numpy()

        states = np.concatenate([
            states,
            np.zeros(shape=(self.width,), dtype=states.dtype),
        ])

        # Initialize the output array with NaN values
        output_array = np.full(
            (len(self.unique_rowids), len(self.unique_positions)),
            np.nan
        )

        # Populate the output array directly using the indices
        output_array[self.row_indices, self.position_indices] = states

        return output_array.astype(float)


    def methylated_cpg(self):
        """1 where CpG is methylated, null_grey otherwise"""
        cpg_met = pc.equal(self.window.locus_data.methylation, pa.scalar(1))
        image = self._protoimage(cpg_met)
        image[image==0.] = self.null_grey

        return self.finish_image(image)

    def methylated_other(self):
        """1 where non-cpg methylated, null_grey otherwise"""
        other_met = pc.greater(self.window.locus_data.methylation, pa.scalar(1))
        image = self._protoimage(other_met)
        image[image == 0.] = self.null_grey

        return self.finish_image(image)


    def methylated_any(self):
        """1 where non-cpg methylated, null_grey otherwise"""
        other_met = pc.greater(self.window.locus_data.methylation, pa.scalar(0))
        image = self._protoimage(other_met)
        image[image == 0.] = self.null_grey

        return self.finish_image(image)


    def bases(self):
        """Shades of grey for each colour"""
        image = self._protoimage(
            self.window.locus_data.nucleotide
        )
        # gives values (0.4, 0.6, 0.8, 1.)
        image = image/5 + 0.2

        return self.finish_image(image)


    def bases_met_as_fith(self):
        """The normal 4 plus methylated C"""
        image = self._protoimage(
            self.window.locus_data.nucleotide
        )

        mask = self.methylated_any() == 1
        image[mask] = 5

        image = image/6 + 1/6

        return self.finish_image(image)


    def strand(self):
        """light grey for rev strand, white for for strand."""
        image = self.read_id_image_copy
        for rid in self.unique_rowids:
            isfor = self.window.read_data.is_forward[rid]
            c = self.strand_colours[isfor == TruePA]
            image[image==rid] = c

        return self.finish_image(image)


    def mapping_quality(self):
        """Each read a shade of grey proportional to mapping quality"""
        image = self.read_id_image_copy
        for rid in self.unique_rowids:
            mapq = self.window.read_data.mapping_quality[rid]

            mapq += self.null_grey
            mapq /= self.window.read_data.max_mapq + self.null_grey

            image[image==rid] = mapq

        return self.finish_image(image)


    def phred(self):
        image = self._protoimage(self.window.locus_data.phred_scores)
        image /= self.window.locus_data.max_phred

        return self.finish_image(image)


    def fill_reads_one_colour(self, c:float):
        image = self._protoimage(c)

        return self.finish_image(image)



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

    bam = AlignmentFile(bamfn)
    if not bam.header.get('HD', {}).get('SO', 'Unknown') == 'coordinate':
        raise RuntimeError(f'BAM file must be sorted by coordinate')

    for aln in iter_bam(bam, paired_end=paired):
        aln:Alignment

        hit_regns = aln.hit_regions(regions)

        if hit_regns:
            n_reads += 1
            n_bases += aln.a.query_length

    # this dtype has implications...
    position_dtype = np.uint32

    read_arrays = dict(
        readID=np.zeros(dtype=np.uint32, shape=(n_reads,)),
        start=np.zeros(dtype=position_dtype, shape=(n_reads,)),
        end=np.zeros(dtype=position_dtype, shape=(n_reads,)),
        chrm=np.zeros(dtype=np.uint8, shape=(n_reads,)),
        mapping_quality=np.zeros(dtype=np.uint8, shape=(n_reads,)),
        is_forward=np.zeros(dtype=np.uint8, shape=(n_reads,))
    )
    loci_arrays = dict(
        nucleotide=np.zeros(dtype=np.uint8, shape=(n_bases,)),
        phred_scores=np.zeros(dtype=np.uint8, shape=(n_bases,)),
        chrm=np.zeros(dtype=np.uint8, shape=(n_bases,)),
        position=np.zeros(dtype=np.uint32, shape=(n_bases,)),
        readID=np.zeros(dtype=np.uint32, shape=(n_bases,)),
        methylation=np.zeros(dtype=np.uint8, shape=(n_bases,)),
        is_insertion=np.zeros(dtype=np.uint8, shape=(n_bases,))
    )

    print('After:')
    print_memory_footprint()

    max32 = np.iinfo(position_dtype).max
    read_i = -1
    position_i_next = 0
    alignmentfile = pysam.AlignmentFile(bamfn)
    for readID, aln in enumerate(iter_bam(alignmentfile, paired_end=paired)):
        aln:Alignment

        if not aln.hit_regions(regions):
            continue

        # deal with the table indicies
        read_i += 1
        position_i = position_i_next
        position_j = position_i + aln.a.query_length
        position_i_next = position_j

        read_arrays['readID'][read_i] = readID
        read_arrays['mapping_quality'][read_i] = aln.a.mapping_quality
        read_arrays['start'][read_i] = aln.a.reference_start
        read_arrays['end'][read_i] = aln.a.reference_end
        read_arrays['chrm'][read_i] = aln.a.reference_id
        read_arrays['is_forward'][read_i] = aln.a.is_forward
        #read_arrays['read_name'][read_i] = a.a.query_name

        loci_arrays['nucleotide'][position_i:position_j] = encode_nt_str(aln.a.query_sequence, )
        loci_arrays['phred_scores'][position_i:position_j] = np.array(aln.a.query_qualities, dtype=np.uint8)
        loci_arrays['chrm'][position_i:position_j] = chrm_map[aln.a.reference_name]
        ref_p = aln.a.get_reference_positions(full_length=True)
        ref_p = np.array([position_dtype(n) if n is not None else max32 for n in ref_p])
        loci_arrays['position'][position_i:position_j] = ref_p
        loci_arrays['readID'][position_i:position_j] = readID
        loci_arrays['methylation'][position_i:position_j] = encode_metstr_bismark(aln.metstr)
        loci_arrays['is_insertion'][position_i:position_j] = ref_p ==  max32

    pa_read_arrays = {
        'readID': pa.array(read_arrays['readID']),
        'start': pa.array(read_arrays['start']),
        'end': pa.array(read_arrays['end']),
        'chrm': pa.array(read_arrays['chrm']),
        'mapping_quality': pa.array(read_arrays['mapping_quality']),
        'is_forward': pa.array(read_arrays['is_forward'], type=pa.bool_()),
    }
    # get things out of memory asap
    del read_arrays

    read_metadata = ReadMetadata(
        pa.Table.from_pydict(pa_read_arrays)
    )
    del pa_read_arrays

    pa_loci_arrays = {
        'position': pa.array(loci_arrays['position']),
        'chrm': pa.array(loci_arrays['chrm']),
        'nucleotide': pa.array(loci_arrays['nucleotide']),
        'phred_scores': pa.array(loci_arrays['phred_scores']),

        'readID': pa.array(loci_arrays['readID']),
        'methylation': pa.array(loci_arrays['methylation']),
        'is_insertion': pa.array(loci_arrays['is_insertion'], type=pa.bool_())
    }
    print('max point probably')
    print_memory_footprint()
    del loci_arrays

    loci_table = pa.Table.from_pydict(pa_loci_arrays)
    loci_table = loci_table.sort_by([
        ('chrm', 'ascending'),
        ('position', 'ascending')
    ])

    loci_data = LocusData(
        loci_table,
        chrm_map=chrm_map,
    )
    print('Just before returning')
    print_memory_footprint()

    print('Reads: ', read_i, 'Locus: ', position_i_next)

    logger.info("Removing insertions...")
    loci_data.remove_insertions()

    return ReadData(loci_data, read_metadata, bam_fn=bamfn,
                    bam_header=str(alignmentfile.header))


def ttests(test_bam_fn, test_regions_fn, testoutdir,
           *, delete_first=False):
    import datetime
    testoutdir = Path(testoutdir)
    if delete_first:
        for f in os.listdir(testoutdir):
            os.remove(testoutdir/f)
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

    window = rd.window(970536, 970586, '1')
    img = Image(window)
    print(img.methylated_cpg())




if __name__ == '__main__':

    bm = '/home/jcthomas/DevLab/NIMBUS/Data/test/bismark_10k.bam'
    #bm = '/home/jcthomas/data/canary/sorted_qname/CMDL19003173_1_val_1_bismark_bt2_pe.deduplicated.bam'
    rg = '/home/jcthomas/DevLab/NIMBUS/Reference/regions-table.canary.4k.tsv'
    out = '/home/jcthomas/DevLab/NIMBUS/Data/test/readdata_structure_test'
    ttests(bm, rg, out, delete_first=True)