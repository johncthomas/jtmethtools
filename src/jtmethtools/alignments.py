import typing
from typing import Collection, Tuple, Literal, Self, Iterable
from numpy.typing import NDArray

import pandas as pd
import numpy as np
import pysam
from loguru import logger
from pysam import AlignedSegment, AlignmentFile, AlignmentHeader
from attrs import define, field
from functools import cached_property

logger.remove()

SplitTable = dict[str, pd.DataFrame]

from jtmethtools.util import (
    load_region_bed,
    split_table_by_chrm,
)

from pathlib import Path

Pathy = str|Path

@define(slots=True)
class LociRange:
    start:int
    end:int
    chrm:str
    name:str=None

    def to_kwargs(self):
        """dict with keys chrm, start, end."""
        return dict(chrm=self.chrm, start=self.start, end=self.end)

    def to_tuple(self):
        """Returns: (start, end, chrm)"""
        return (self.start, self.end, self.chrm)


@define
class Regions:
    """Region starts, ends and names stored in vectors
    as attributes of the same names.

    Create using Regions.from_file or .from_df
    """
    starts: dict[str, NDArray[int]]
    ends: dict[str, NDArray[int]]
    names: dict[str, NDArray[str]]
    thresholds: dict[str, float] = None
    df: pd.DataFrame = None

    def iter(self) -> typing.Iterable[LociRange]:
        for _, row in self.df.iterrows():
            yield LociRange(
                start=row.Start,
                end=row.End,
                chrm=row.Chrm,
                name=row.Name
            )

    @cached_property
    def chromsomes(self) -> set[str]:
        return set(self.df.Chrm.unique())

    @classmethod
    def from_file(cls, filename: Pathy) -> Self:
        filename = str(filename)
        if filename.endswith('.bed') or filename.endswith('.txt'):
            return cls.from_bed(filename)
        df = pd.read_csv(filename, sep='\t', dtype={'Chrm':str})
        df.set_index( 'Name', inplace=True, drop=False)
        return (cls.from_df(df))


    @classmethod
    def from_bed(cls, filename: Pathy) -> Self:
        raise NotImplementedError("Table now needs thresholds.")
        df = load_region_bed(filename)
        return cls.from_df(df)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        sdf = split_table_by_chrm(df)

        return cls(
            starts={k: sdf[k].Start.values for k in sdf},
            ends={k: sdf[k].End.values for k in sdf},
            names={k: sdf[k].Name.values for k in sdf},
            thresholds=df.Threshold.to_dict() if 'Threshold' in df.columns.values else None,
            df=df
        )

    def starts_ends_of_chrm(self, chrm) -> (NDArray[int], NDArray[int]):
        return (self.starts[chrm], self.ends[chrm])

    def get_region_threshold(self, name):
        return self.thresholds[name]

def alignment_overlaps_region(
        alignment: AlignedSegment,
        regions: Regions) -> bool | str:
    """Check if an alignment overlaps a region."""
    ref = alignment.reference_name
    try:
        regStart, regEnd = regions.starts_ends_of_chrm(ref)
    except KeyError:
        logger.trace('Reference contig not found: ' + str(alignment.reference_name))
        return False

    m = (
            (alignment.reference_start < regEnd)
            & (alignment.reference_end > regStart)
    )

    isoverlap = np.any(m)

    if isoverlap:
        reg_name = regions.names[ref][m][0]
        logger.trace(reg_name)
        return reg_name #its fine
    return False


def get_bismark_met_str(a: AlignedSegment) -> str:
    tag, met = a.get_tags()[2]
    try:
        assert tag == 'XM'
    except AssertionError:
        tags = a.get_tags()
        for t, m in tags:
            if t == 'XM':
                tag, met = t, m
        if tag != 'XM':
            raise ValueError(f"Can't find XM tag in alignment {a.query_name}")
    return met

def write_bam_from_pysam(
        out_fn:str,
        alignments:Collection[pysam.AlignedSegment],
        header_file:str=None,
        header:AlignmentHeader=None,
):
    """Write a bam file of alignments. Supply a SAM/BAM file name to
    use for the header, or supply the header directly. Either
    `header_file` or `header`, not both.

    Output format (BAM or SAM) determined from out_fn.
    """
    if (
           ( (header_file is None) and (header is None) )
        or ( (header_file is not None) and (header is not None) )
    ):
        raise ValueError("One of either `header_file` or `header` must be supplied.")

    if header_file is not None:
        header = AlignmentFile(header_file).header

    if str(out_fn).lower().endswith('.bam'):
        mode = 'wb'
    elif str(out_fn).lower().endswith('.sam'):
        mode = 'w'
    else:
        raise ValueError(f"Can't determine output file type from file name {out_fn}. "
                         f"Needs to end with .sam or .bam.")
    with AlignmentFile(out_fn, mode=mode, header=header) as out:
        for a in alignments:
            if a is not None:
                out.write(a)


def flag_to_text(flag):
    """
    Convert the integer flag from pysam AlignedSegment to a human-readable text representation.
    """
    # Flag definitions according to the SAM format specification
    flag_descriptions = {
        0x1: "Paired read",
        0x2: "Properly paired",
        0x4: "Unmapped read",
        0x8: "Mate unmapped",
        0x10: "Read reverse strand",
        0x20: "Mate reverse strand",
        0x40: "First in pair",
        0x80: "Second in pair",
        0x100: "Not primary alignment",
        0x200: "Read fails platform/vendor quality checks",
        0x400: "PCR or optical duplicate",
        0x800: "Supplementary alignment"
    }

    result = []

    # Check each flag bit and append the description if the bit is set
    for flag_value, description in flag_descriptions.items():
        if flag & flag_value:
            result.append(description)

    return ", ".join(result) if result else "None"


def get_alignment_of_read(readname:str, bamfn:str) -> (AlignedSegment, AlignedSegment):
    import pysam
    with pysam.AlignmentFile(bamfn) as bam:

        aligns = []
        for a in bam:
            if a.qname == readname:
                aligns.append(a)
        return tuple(aligns)


def get_ref_position(seq_i, ref_start, cigar):
    """Get the reference locus, taking insertions/deletions into account"""
    ref_pos = ref_start
    seq_pos = 0

    for (cigar_op, cigar_len) in cigar:
        if cigar_op in (0, 7, 8):  # M, =, X: consume both ref and query
            if seq_pos + cigar_len > seq_i:
                return ref_pos + (seq_i - seq_pos)
            seq_pos += cigar_len
            ref_pos += cigar_len
        elif cigar_op == 1:  # I: consume query only
            if seq_pos + cigar_len > seq_i:
                return None  # insertion before current position
            seq_pos += cigar_len
        elif cigar_op == 2:  # D: consume ref only
            ref_pos += cigar_len
        elif cigar_op == 3:  # N: skip region (introns typically)
            ref_pos += cigar_len
        elif cigar_op == 4:  # S: soft clipping, consume query only
            if seq_pos + cigar_len > seq_i:
                return None  # soft clip before current position
            seq_pos += cigar_len
        elif cigar_op == 5:  # H: hard clipping, consume nothing
            continue
        elif cigar_op == 6:  # P: padding, consume neither
            continue

    return None  # position not reached, or it's past the alignment


@define(frozen=True)
class Alignment:
    a: AlignedSegment
    a2: AlignedSegment | None = None
    kind: Literal['bismark'] = 'bismark'
    filename: str = ''

    def _get_met_str(self, a:AlignedSegment):
        if self.kind == 'bismark':
            return get_bismark_met_str(a)

    @cached_property
    def _a1_a2_overlaplen(self):
        """Returns alignments in position order and length of the overlap"""


        a1, a2 = self.a, self.a2

        # Put a1 on the left of a2
        if a1.reference_start > a2.reference_start:
            a2, a1 = a1, a2


        overlap_length = a1.reference_end - a2.reference_start
        return a1, a2, overlap_length

    @cached_property
    def metstr(self) -> str:
        if self.a2 is None:
            return self._get_met_str(self.a)
        a1, a2, overlap_length = self._a1_a2_overlaplen
        m1, m2 = [self._get_met_str(a) for a in (a1, a2)]

        # no overlap, just concat
        if overlap_length < 1:
            newm = m1 + m2
        else:
            # remove overlapping region from m1 and add m2
            newm = m1[:-overlap_length] + m2
        return newm


    @cached_property
    def alignments(self) -> Tuple[AlignedSegment]:
        if self.a2 is None:
            alignments = (self.a,)
        else:
            alignments = (self.a, self.a2)
        return alignments

    @cached_property
    def reference_start(self):
        if self.a2 is None:
            return self.a.reference_start
        else:
            return min(self.a.reference_start, self.a2.reference_start)

    @cached_property
    def reference_end(self):
        if self.a2 is None:
            return self.a.reference_end
        else:
            return max(self.a.reference_end, self.a2.reference_end)

    def mapping_quality(self, keep_lowest=True):
        if self.a2 is None:
            return self.a.mapping_quality
        else:
            if keep_lowest:
                minmax = min
            else:
                minmax = max
            return minmax(self.a.mapping_quality, self.a2.mapping_quality)

    def _iter_locus_metstr_bismark(self, a: AlignedSegment) -> Tuple[int, bool]:
        """Iterate through the bismark methylation string, yielding
        the chromosomal position of current CpG and it's methylation
        state (True if methylated)
        """
        met_str = self._get_met_str(a)
        for i, m in enumerate(met_str):
            if m in 'zZ':
                locus = get_ref_position(i, a.reference_start, a.cigartuples)

                if locus is None:
                    continue
                locus += 1
                yield locus, m == 'Z'

    @cached_property
    def locus_methylation(self) -> dict[str, bool]:
        """Dict of locus->True|False indicating a methylated CpG.

        If alignment is a paired end, overlapping loci are merged."""
        a1, a2 = self.a, self.a2
        d1 = {k: l for k, l in self._iter_locus_metstr_bismark(a1)}
        if a2 is None:
            return d1
        d2 = {k: l for k, l in self._iter_locus_metstr_bismark(a2)}
        return d1 | d2

    def no_non_cpg(self) -> bool:
        """look for forbidden methylation states.

        Return True no H|X."""
        values = set(self.metstr)
        if (
                ('H' in values)
                or ('X' in values)
                or ('U' in values)
        ):
            return False
        return True

    def get_hit_regions(self, regions:Regions) -> list[str]:
        regions = [alignment_overlaps_region(a, regions)
                   for a in self.alignments]
        regions = list(set([r for r in regions if r]))
        return regions

def _load_bam(bam:str|Path|AlignmentFile):
    if not isinstance(bam, AlignmentFile):
        mode = 'rb'
        if str(bam).endswith('.sam'):
            mode = 'r'
        bam = pysam.AlignmentFile(bam, mode)
    return bam

def _iter_bam_pe_qname_sorted(
        bam:str|Path|AlignmentFile,
) -> Iterable[Tuple[AlignedSegment, AlignedSegment|None]]:
    """Iterate over a paired-end bam file, yielding pairs of alignments.
    Where a read is unpaired, yield (alignment, None).
    """

    bam = _load_bam(bam)

    if not bam.header.get('HD', {}).get('SO', 'Unknown') == 'queryname':
        raise RuntimeError(f'BAM file must be sorted by queryname')

    aln_prev: pysam.AlignedSegment | None = None
    for i, aln_current in enumerate(bam):
        logger.debug(f'Alignment #{i}')


        if aln_prev is None:
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


def _iter_pe_bam_unsorted(
        bam:AlignmentFile
) -> Iterable[Tuple[AlignedSegment, AlignedSegment | None]]:
    alignment_buffer = {}
    for a in bam:
        qn = a.query_name
        if qn not in alignment_buffer:
            alignment_buffer[qn] = a
        else:
            a1, a2 = a, alignment_buffer[qn]
            del alignment_buffer[qn]
            yield a1, a2
    if alignment_buffer:
        for a in alignment_buffer.values():
            yield a, None


def _iter_bam_se(
        bam:str|Path|AlignmentFile,
) -> Iterable[AlignedSegment]:
    """Iterate over a single-ended bam file, yielding pairs of alignments.
    Where a read is unpaired, yield (alignment, None).

    Use start_stop for splitting a bam file for, e.g. multiprocessing.
    """

    bam = _load_bam(bam)

    for i, aln in enumerate(bam):
        logger.debug(f'Alignment #{i}')

        yield aln


def iter_bam(
        bam:str|Path|AlignmentFile,
        paired_end:bool=True,
        kind='bismark',
) -> Iterable[Alignment]:
    """Iterate over a bam file, yielding Alignments.
    """
    bam = _load_bam(bam)
    sorting_method = bam.header.get('HD', {}).get('SO', 'Unknown')

    if paired_end:
        if sorting_method == 'queryname':
            bamiter = _iter_bam_pe_qname_sorted(bam )
        else:
            logger.info(
                "Paired end bam not sorted by queryname - this might take a little longer and use more memory.\n"
                "If it's actually single-ended it'll take a lot more memory. Use the --single-ended option to avoid this.\n"
                f"If it's paired end and sorted by query name it shouldn't be too bad... ({sorting_method:})"
            )
            bamiter = _iter_pe_bam_unsorted(bam)
        for aln in bamiter:
            yield Alignment(*aln, kind=kind)
    else:
        bamiter = _iter_bam_se(bam, )
        for aln in bamiter:
            yield Alignment(aln, kind=kind)
    return None


