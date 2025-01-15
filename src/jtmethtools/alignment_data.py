import json
import pathlib
from os import PathLike
import os
from pathlib import Path
from typing import (
    Self,
    Iterator,
    Iterable
)

from functools import lru_cache

import attrs

import numpy as np

from numpy.typing import NDArray

import pysam
from pysam import AlignedSegment, AlignmentFile

import pyarrow as pa
import pyarrow.compute as compute
from pyarrow import parquet

from attrs import define

from jtmethtools.alignments import (
    Alignment,
    iter_bam,
    Regions,
    LociRange
)

from jtmethtools.util import (
    logger
)

logger.remove()

Pathesque = str | Path | PathLike[str]

# pyarrow bools aren't recognised as bools by python, they
# are Truthy, even when `false`. So using x == TruePA
TruePA = pa.scalar(True, type=pa.bool_())
FalsePA = pa.scalar(False, type=pa.bool_())

_ntcodes = dict(zip('ACGTN', np.array([1, 2, 3, 4, 0], dtype=np.uint8)))
NT_CODES = _ntcodes | {v:k for k, v in _ntcodes.items()}

_bsmk_codes = dict(zip('.ZHXU', np.array([0, 1, 2, 3, 4], dtype=np.uint8)))
BISMARK_CODES = _bsmk_codes | {v:k for k, v in _bsmk_codes.items()}

def _encode_string(string:Iterable[str], codes:dict) -> NDArray[np.uint8]:
    """Perform substitution of string to values in codes."""
    nt_arr = np.zeros((len(string),), dtype=np.uint8)
    for i, n in enumerate(string):
        try:
            nt_arr[i] = codes[n]
        except KeyError:
            pass
    return nt_arr


def encode_nt_str(nts:Iterable[str]) -> NDArray[np.uint8]:
    return _encode_string(nts, NT_CODES)


def encode_metstr_bismark(metstr:Iterable[str]) -> NDArray[np.uint8]:
    return _encode_string(metstr, BISMARK_CODES)


@define
class Column:
    name:str
    pa_array_type:type
    pa_dtype:type
    np_dtype:type

# column types used for keeping equivilent array dtypes together
#  in the code.
POS_NP_DTYPE = np.uint32
@define
class PositionCol(Column):
    name:str
    pa_array_type:type=pa.UInt32Array
    pa_dtype:type=pa.UInt32Scalar
    np_dtype:type=POS_NP_DTYPE

class COLS_READ:
    readID = Column('readID', pa.UInt32Array, pa.UInt32Scalar, np.uint32)
    start = PositionCol('start')
    end = PositionCol('end')
    chrm = Column('chrm', pa.UInt8Array, pa.UInt8Scalar, np.uint8)
    mapping_quality = Column('mapping_quality', pa.UInt8Array, pa.UInt8Scalar, np.uint8)
    is_forward = Column('is_forward', pa.BooleanArray, pa.BooleanScalar, np.bool_)
    read_name = Column('read_name', pa.StringArray, pa.StringScalar, object)

class COLS_LOCUS:
    readID = Column('readID', pa.UInt32Array, pa.UInt32Scalar, np.uint32)
    nucleotide = Column('nucleotide', pa.UInt8Array, pa.UInt8Scalar, np.uint8)
    phred_scores = Column('phred_scores', pa.UInt8Array, pa.UInt8Scalar, np.uint8)
    position = PositionCol('position')
    methylation = Column('methylation', pa.UInt8Array, pa.UInt8Scalar, np.uint8)
    is_insertion = Column('is_insertion', pa.BooleanArray, pa.BooleanScalar, np.bool_)
    chrm = Column('chrm', pa.UInt8Array, pa.UInt8Scalar, np.uint8)
    is_cpg = Column('is_cpg', pa.BooleanArray, pa.BooleanScalar, np.bool_)


def _iter_col(columns):
    for k in dict(vars(columns)):
        if k.startswith('_'):
            continue
        col:Column = getattr(columns, k)
        yield col
def iter_read_cols() -> Iterator[Column]:
    for col in _iter_col(COLS_READ):
        yield col
def iter_locus_cols() -> Iterator[Column]:
    for col in _iter_col(COLS_LOCUS):
        yield col


PositionArray = PositionCol.pa_array_type
ReadIDArray = COLS_LOCUS.readID.pa_array_type
NucleotideArray = COLS_LOCUS.nucleotide.pa_array_type
PhredArray = COLS_LOCUS.phred_scores.pa_array_type
MethylationArray = COLS_LOCUS.methylation.pa_array_type
ChrmArray = COLS_LOCUS.chrm.pa_array_type


class ReadTable:
    __slots__ = ('table', 'max_mapq')
    def __init__(
            self,
            table:pa.Table,
            max_mapq=42,
    ):
        """Table holds read level information, columns are
        accessible as properties of this object."""

        self.table = table
        self.max_mapq = max_mapq
        if compute.greater(
                compute.max(self.mapping_quality),
                pa.scalar(max_mapq, type=pa.uint8())
        ) == TruePA:
            max_found = compute.max(self.mapping_quality).as_py()
            raise RuntimeError(
                f"Max MAPQ exceeded ({max_found} > {max_mapq}) , the alignment is using a different"
                " scale and image gen would need to be rewritten"
            )

    def _nontable_attr(self):
        return dict(max_mapq=self.max_mapq)

    def get_col(self, col:str):
        return self.table.column(col)

    @property
    def readID(self) -> COLS_READ.readID.pa_array_type:
        return self.table.column(COLS_READ.readID.name)

    @property
    def start(self) -> COLS_READ.start.pa_array_type:
        return self.table.column(COLS_READ.start.name)

    @property
    def end(self) ->  COLS_READ.start.pa_array_type:
        return self.table.column(COLS_READ.end.name)

    @property
    def chrm(self) ->  COLS_READ.chrm.pa_array_type:
        return self.table.column(COLS_READ.chrm.name)

    @property
    def mapping_quality(self) ->  COLS_READ.mapping_quality.pa_array_type:
        return self.table.column(COLS_READ.mapping_quality.name)

    @property
    def read_name(self) -> pa.StringArray:
        return self.table.column("read_name")

    @property
    def is_forward(self) -> COLS_READ.is_forward.pa_array_type:
        return self.get_col(COLS_READ.is_forward.name)

    @lru_cache(255)
    def loci_from_readID(self, readID:int) -> LociRange:
        row = self.table.slice(readID, readID)
        return LociRange(
            start=row.column('start'),
            end=row.columns('end'),
            chrm=row.columns('chrm')
        )

    @lru_cache(255)
    def read_ids_at_loci(self, start:int, end:int, chrm:int) -> PositionArray:
        condition1 = compute.less_equal(self.table.column("start"), pa.scalar(end))
        condition2 = compute.greater_equal(self.table.column("end"), pa.scalar(start))
        condition3 = compute.equal(self.table.column("chrm"), pa.scalar(chrm))

        # combine the conditions
        combined_condition = compute.and_(compute.and_(condition1, condition2), condition3)

        # Filter the table to get rows where all conditions are true
        filtered_readIDs = compute.filter(self.table.column("readID"), combined_condition)

        # Return the filtered readIDs (which will be a pyarrow.Array)
        return filtered_readIDs

    def get_value(self, read_id:int, col:str) -> pa.Table:
        t = self.readID
        m = compute.equal(t, pa.scalar(read_id, type=pa.uint32()))
        v = self.table.column(col).filter(m).to_pylist()
        logger.trace(f"readid: {read_id}")
        logger.trace(v)
        return v[0]

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

    def filter(self, *args, **kwargs) -> Self:
        """Args passed to self.table.filter(), returns LocusData with
        filtered self.table"""
        return ReadTable(self.table.filter(*args, **kwargs),
                         **self._nontable_attr())



@define
class LocusTable:

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
    def position(self) -> PositionArray:
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

    @property
    def is_cpg(self) -> pa.BooleanArray:
        return self.table.column('is_cpg')

    def to_parquet(self, fn:Path|str):
        """Write table in Parquet format"""
        parquet.write_table(self.table, fn)

    def loci_mask(self, loci:LociRange) -> pa.BooleanArray:
        loc_m = compute.and_(
            compute.greater_equal(
                self.position,
                pa.scalar(loci.start)
            ),
            compute.less(
                self.position,
                pa.scalar(loci.end)
            )
        )

        chrm_m = compute.equal(
            self.chrm,
            pa.scalar(self.chrm_map[loci.chrm])
        )

        return compute.and_(loc_m, chrm_m)


    def filter(self, *args, **kwargs) -> Self:
        """Args passed to self.table.filter(), returns LocusData with
        filtered self.table"""
        return LocusTable(self.table.filter(*args, **kwargs), chrm_map=self.chrm_map)

    def filter_by_readID(
            self,
            read_ids:COLS_LOCUS.readID.pa_array_type,
            remove_ids=False) -> Self:

        m = compute.is_in(self.readID, read_ids)
        if remove_ids:
            m = compute.invert(m)
        return self.filter(m)

    def remove_insertions(self) -> Self:
        return self.filter(compute.invert(self.is_insertion))

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

    # @classmethod
    # def from_parquet(cls, fn:str|Path):
    #     """Read table from Parquet format"""
    #     return cls(parquet.read_table(fn))

    def count_cpg_per_read(self):
        """Count the number of CpGs in each read"""

        cpg_counts = self.table.group_by(COLS_LOCUS.readID.name).aggregate(
            [(COLS_LOCUS.is_cpg.name, 'sum')]
        )

        return cpg_counts

    def filter_cpg_per_read(self, min_cpg:int) -> Self:
        """Remove rows where number of CpG in read < min_cpg"""
        min_cpg = pa.scalar(min_cpg)

        cpg_counts = self.count_cpg_per_read()

        # get readID where is_cpg > min_cpg
        read_ids = cpg_counts.filter(
            compute.greater_equal(cpg_counts.column('is_cpg_sum'), min_cpg)
        ).column('readID')

        # get rows where readID is in read_ids
        return self.filter(compute.is_in(self.readID, read_ids))

    def filter_noncpg_met_per_read(self, max_noncpg=0):
        # filter by methylation type
        noncpg_mask = compute.greater_equal(
            self.methylation,
            pa.scalar(2)
        )

        # count freq noncpg per read
        noncpg = self.readID.filter(noncpg_mask)
        # returns a struct with fields "counts", "values"
        readid_count = compute.value_counts(noncpg)

        m = compute.greater(
            readid_count.field('counts'),
            pa.scalar(max_noncpg)
        )

        bad_rids = readid_count.field('values').filter(m)

        return self.filter_by_readID(bad_rids, remove_ids=True)





@define
class AlignmentsData:
    locus_data:LocusTable
    read_data:ReadTable
    bam_fn:str
    bam_header:str
    version:int = 1 #increment when attributes change
    max_phred=42

    def _get_nontable_attr(self):
        d = attrs.asdict(self)
        del d['locus_data']
        del d['read_data']
        d['bam_fn'] = str(d['bam_fn'])
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

        with open(directory/'short-fields.json', 'w') as f:
            json.dump(d, f)

    @classmethod
    def from_dir(cls, directory:Pathesque):
        directory = pathlib.Path(directory)
        loci = LocusTable.from_dir(directory)
        read = ReadTable.from_dir(directory)

        with open(directory/'short-fields.json', 'r') as f:
            d = json.load(f)

        return cls(loci, read, **d)

    @staticmethod
    def from_bam(
            bamfn:Pathesque,
            regionsfn:Pathesque|None=None
    ):
        self = process_bam(bamfn, regionsfn)
        return self

    def print_heads(self, n=5) -> None:
        logger.info('Read table:')
        logger.info(self.read_data.table.slice(0, n))
        logger.info('\n++++\n')
        logger.info('Loci table:')
        logger.info(self.locus_data.table.slice(0, n))


    def loci_mask(self, loci:LociRange) -> pa.BooleanArray:
        return self.locus_data.loci_mask(loci)

    def window(
            self,
            loci:LociRange=None,
            *,
            start:int=None,
            end:int=None,
            chrm:str=None
    ) -> Self:
        """Return an AlignmentData subset to the given loci range.

        Pass loci object, OR start/end/chrm as keywords."""
        if loci is None:
            loci = LociRange(
                start=start,
                end=end,
                chrm=chrm
            )
        mask = self.loci_mask(loci)
        locuswindow = self.locus_data.filter(mask)
        # note: at the moment can't filter read_data cus we rely on the row
        # positions, but it also should be relatively small.
        return AlignmentsData(locuswindow, self.read_data, **self._get_nontable_attr())


    def filter_by_ncpg(self, min_cpg) -> Self:
        """Remove reads where read contains < min_cpg"""
        locus_data = self.locus_data.filter_cpg_per_read(min_cpg)
        # keep reads still in locus_data.readID
        read_data = self.read_data.filter(
            compute.is_in(self.read_data.readID, value_set=locus_data.readID)
        )
        return AlignmentsData(locus_data, read_data,
                              **self._get_nontable_attr())


    def filter_by_mapping_quality(self, min_mapq:int) -> Self:
        """Remove reads where mapping_quality < min_mapq"""
        read_data = self.read_data.filter(
            compute.greater_equal(self.read_data.mapping_quality, pa.scalar(min_mapq))
        )

        locus_data = self.locus_data.filter_by_readID(read_data.readID)

        return AlignmentsData(
            locus_data, read_data,
            **self._get_nontable_attr()
        )


    def filter_by_noncpg_met(self, max_noncpg=0):
        """Remove reads with methylated non-CpG cytosines more than
        max_noncpg"""
        locus_data = self.locus_data.filter_noncpg_met_per_read(max_noncpg)
        read_data = self.read_data.filter(
            compute.is_in(self.read_data.readID, locus_data.readID)
        )

        return AlignmentsData(locus_data, read_data,
                              **self._get_nontable_attr())



# read ID used for padding row in generation of images
#  hopefully shouldn't exist in teh actual data cus we want
#  it to crash if it's ever accidentally used to look up info.
PAD_READID = np.iinfo(POS_NP_DTYPE).max


def get_insertion_mask(a:AlignedSegment,):
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

    logger.info(f"Memory usage: {memory_usage_mb:.2f} MB")


def process_bam(bamfn, regionsfn:str|Path,
                filter_by_region=True,
                include_read_name=False,
                single_ended=False) -> AlignmentsData:
    regionsfn = str(regionsfn)
    # get the number of reads and the number of aligned bases
    n_reads = 0
    n_bases = 0
    paired = not single_ended

    if regionsfn.endswith('.tsv'):
        regions = Regions.from_file(regionsfn)
    else:
        regions = Regions.from_bed(regionsfn)

    def check_hits_region() -> bool:
        if filter_by_region:
            return len(aln.get_hit_regions(regions)) > 0
        else:
            return True

    chrm_map = {}
    for i, c in enumerate(regions.chromsomes):
        chrm_map[i] = c
        chrm_map[c] = i

    logger.info('\n')
    logger.info(chrm_map)

    logger.info('Before creating creating empty arrays:')
    print_memory_footprint()

    bam = AlignmentFile(bamfn)

    logger.info('Counting number of reads that hit a region')
    for i, aln in enumerate(iter_bam(bam, paired_end=paired)):
        aln:Alignment

        if check_hits_region():
            n_reads += 1
            n_bases += len(aln.metstr)

    logger.info(f"{n_reads} of {i} reads hit a region {n_reads / i * 100:.2f}%")

    read_arrays = {}
    for col in iter_read_cols():
        col:Column
        read_arrays[col.name] = np.zeros(
            dtype=col.np_dtype,
            shape=(n_reads,)
        )
    loci_arrays = {}
    for col in iter_locus_cols():
        col: Column

        loci_arrays[col.name] = np.zeros(
            dtype=col.np_dtype,
            shape=(n_bases,)
        )

    logger.info('After creating empty arrays:')
    print_memory_footprint()

    max_pos = np.iinfo(POS_NP_DTYPE).max
    read_i = -1
    position_i_next = 0
    alignmentfile = pysam.AlignmentFile(bamfn)
    for readID, aln in enumerate(iter_bam(alignmentfile, paired_end=paired)):
        aln:Alignment

        if not check_hits_region():
            continue

        # deal with the table indicies
        read_i += 1
        position_i = position_i_next
        position_j = position_i + len(aln.metstr)
        position_i_next = position_j

        read_arrays['readID'][read_i] = readID
        read_arrays['mapping_quality'][read_i] = aln.mapping_quality()
        read_arrays['start'][read_i] = aln.reference_start
        read_arrays['end'][read_i] = aln.reference_end
        read_arrays['chrm'][read_i] = aln.a.reference_id
        read_arrays['is_forward'][read_i] = aln.a.is_forward
        if include_read_name:
            read_arrays['read_name'][read_i] = aln.a.query_name

        # per locus data
        locus_data = aln.locus_values
        encoded_nt = encode_nt_str(locus_data.nucleotides.values(), )
        loci_arrays['nucleotide'][position_i:position_j] = encoded_nt
        loci_arrays['phred_scores'][position_i:position_j] = np.array(
            list(locus_data.qualities.values()), dtype=np.uint8
        )
        loci_arrays['chrm'][position_i:position_j] = chrm_map[aln.a.reference_name]
        ref_p = np.array(
            list(locus_data.nucleotides.keys()),
            dtype=POS_NP_DTYPE
        )
        loci_arrays['position'][position_i:position_j] = ref_p
        loci_arrays['readID'][position_i:position_j] = readID
        met_encoded = encode_metstr_bismark(aln.metstr)
        loci_arrays['methylation'][position_i:position_j] = met_encoded
        s = list(aln.metstr.lower())
        iscpg =  np.array(list(aln.metstr.lower())) == 'z'
        loci_arrays['is_cpg'][position_i:position_j] = iscpg
        loci_arrays['is_insertion'][position_i:position_j] = ref_p ==  max_pos

    if not include_read_name:
        del read_arrays['read_name']

    pa_read_arrays = {
        k: pa.array(v) for k, v in read_arrays.items()
    }
    # get things out of memory asap
    del read_arrays

    read_metadata = ReadTable(
        pa.Table.from_pydict(pa_read_arrays, )
    )
    del pa_read_arrays

    pa_loci_arrays = {
        k: pa.array(v) for k, v in loci_arrays.items()
    }
    logger.info('max point probably')
    print_memory_footprint()
    del loci_arrays

    loci_table = pa.Table.from_pydict(pa_loci_arrays)
    # loci_table = loci_table.sort_by([
    #     ('chrm', 'ascending'),
    #     ('position', 'ascending')
    # ])

    loci_data = LocusTable(
        loci_table,
        chrm_map=chrm_map,
    )
    logger.info('Just before returning')
    print_memory_footprint()

    logger.info('Reads: ', read_i, 'Locus: ', position_i_next)

    logger.info("Removing insertions...")
    loci_data.remove_insertions()

    return AlignmentsData(loci_data, read_metadata, bam_fn=bamfn,
                          bam_header=str(alignmentfile.header))


def sampledown_rows(arr:NDArray, max_rows:int) -> NDArray:

    rowi = np.sort(
        np.random.choice(
            list(range(arr.shape[0])),
            size=max_rows,
            replace=False
        )
    )
    return arr[rowi]


# TESTDIR = Path('/home/jcthomas/python/jtmethtools/src/jtmethtools/tests/')
#
# test_bam_fn = TESTDIR / 'img-test.sam'
#
# test_regions_fn = TESTDIR / 'regions.bed'
#
# ttests(test_bam_fn, test_regions_fn, TESTDIR)

# rd = process_bam(
#     test_bam_fn,
#     test_regions_fn
# )
#
# rd.print_heads(100)
#
# windoo = rd.window(start=90, end=110, chrm='1')
# windoo = windoo.filter_by_noncpg_met(0)


# imgee = ImageMaker(windoo, )
# import matplotlib.pyplot as plt
# fig, axes = imgee._plot_test_images()
# plt.savefig(TESTDIR / 'channels.png', dpi=150, )