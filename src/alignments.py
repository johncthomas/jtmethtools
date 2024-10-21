import typing
from typing import Collection, Tuple, Literal

import pandas as pd
import numpy as np
import pysam
from loguru import logger
from pysam import AlignedSegment, AlignmentFile, AlignmentHeader
from attrs import define, field
from functools import cached_property

logger.remove()


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

    @cached_property
    def chromsomes(self) -> set[str]:
        return set(self.df.Chrm.unique())

    @classmethod
    def from_file(cls, filename: str) -> Self:
        df = pd.read_csv(filename, sep='\t',)
        df.set_index( 'Name', inplace=True, drop=False)
        return (cls.from_df(df))


    @classmethod
    def from_bed(cls, filename: str) -> Self:
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

def get_bismark_met_str(a: AlignedSegment) -> str:
    tag, met = a.get_tags()[2]
    assert tag == 'XM'
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


@define(frozen=True)
class Alignment:
    a: AlignedSegment
    a2: AlignedSegment | None = None
    kind: Literal['bismark'] = 'bismark'

    def _get_met_str(self, a:AlignedSegment):
        if self.kind == 'bismark':
            return get_bismark_met_str(a)

    @cached_property
    def metstr(self) -> str:

        if self.a2 is None:
            return self._get_met_str(self.a)

        a1, a2 = self.a, self.a2

        # Put a1 on the left of a2
        if a1.reference_start > a2.reference_start:
            a2, a1 = a1, a2

        m1, m2 = [self._get_met_str(a) for a in (a1, a2)]
        overlap_length = a1.reference_end - a2.reference_start
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



def _iter_bam_pe(
        bam:str|AlignmentFile,
        start_stop:Tuple[int, int]=(0, np.inf)
) -> Tuple[AlignedSegment, AlignedSegment|None]:
    """Iterate over a paired-end bam file, yielding pairs of alignments.
    Where a read is unpaired, yield (alignment, None).

    Use start_stop for splitting a bam file for, e.g. multiprocessing.
    """

    if not isinstance(bam, AlignmentFile):
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


def _iter_bam_se(
        bam:str|AlignmentFile,
        start_stop:Tuple[int, int]=(0, np.inf)
) -> Tuple[AlignedSegment, AlignedSegment|None]:
    """Iterate over a single-ended bam file, yielding pairs of alignments.
    Where a read is unpaired, yield (alignment, None).

    Use start_stop for splitting a bam file for, e.g. multiprocessing.
    """

    if not isinstance(bam, AlignmentFile):
        bam = pysam.AlignmentFile(bam, 'rb')

    for i, aln in enumerate(bam):
        logger.debug(f'Alignment #{i}')
        if i < start_stop[0]:
            continue
        elif i >= start_stop[1]:
            return None

        yield aln



def iter_bam(
        bam:str|AlignmentFile,
        start_stop:Tuple[int, int]=(0, np.inf),
        paired_end:bool=True,
        kind='bismark',
) -> Alignment:
    """Iterate over a bam file, yielding Alignments.

    Use start_stop for splitting a bam file for, e.g. multiprocessing.
    """
    if paired_end:
        bam = _iter_bam_pe(bam, start_stop)
        for aln in bam:
            yield Alignment(*aln, kind=kind)
    else:
        bam = _iter_bam_se(bam, start_stop)
        for aln in bam:
            yield Alignment(aln, kind=kind)
    return None


def _test():
    bamfn = '/home/jcthomas/data/canary/sorted_qname/CMDL19003169_1_val_1_bismark_bt2_pe.deduplicated.bam'

    for alignment in iter_bam(bamfn, (0, 6), paired_end=False):
        meth_by_locus = alignment.locus_methylation
        print(alignment.a.query_name)
        print(alignment.metstr)
        print(meth_by_locus.values())

if __name__ == '__main__':
    logger.remove()
    #_test()