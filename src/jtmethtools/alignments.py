"""Class and functions for working with Bismark BAM."""
import collections
import typing
from typing import Collection, Tuple, Literal, Iterable

import pandas as pd
import numpy as np
import pysam
from loguru import logger
from pysam import AlignedSegment, AlignmentFile, AlignmentHeader
from attrs import define
from functools import cached_property

from jtmethtools.classes import Regions

SplitTable = dict[str, pd.DataFrame]

from pathlib import Path

Pathy = str|Path

ALIGNMENT_ERROR_COUNT = 0

__all__ = [
    'Alignment',
    'iter_bam',
    'iter_bam_segments',
    'get_bismark_met_str',
    'write_bam_from_pysam',
    'alignment_overlaps_region',
    'flag_to_text',
]


def alignment_overlaps_region(
        alignment: AlignedSegment,
        regions: Regions) -> bool | str:
    """Check if an alignment overlaps a region. Returns False if not,
    and the name of the first overlapping region if it does.

    samtools is much faster if you just want to filter a large BAM."""
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


def get_bismark_met_str(a: AlignedSegment) -> str|None:
    """Return the methylation string for a Bismark alignment, or None if it can't be found."""
    # this works in every case I've seen...
    tags = a.get_tags()
    tag = None
    if len(tags) > 2:
        tag, met = a.get_tags()[2]
    try:
        assert tag == 'XM'
    # but it's not guaranteed that the tag position will never change
    #   so this should work whever it is.
    except AssertionError:
        for t, m in tags:
            if t == 'XM':
                tag, met = t, m
                break
        if tag != 'XM':
            logger.warning(f"Can't find XM tag in alignment {a.query_name}, {tags=}")
            met = None

    return met

def write_bam_from_pysam(
        out_fn:str,
        alignments:Collection[pysam.AlignedSegment],
        header_file:str=None,
        header:AlignmentHeader=None,
):
    """Convenience function for writing pysam.AlignedSegments.
    Supply a SAM/BAM file name to
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


# def get_alignment_of_read(readname:str, bamfn:str) -> (AlignedSegment, AlignedSegment):
#     import pysam
#     with pysam.AlignmentFile(bamfn) as bam:
#
#         aligns = []
#         for a in bam:
#             if a.qname == readname:
#                 aligns.append(a)
#         return tuple(aligns)


# @lru_cache(1024)
# def get_ref_position(seq_i, ref_start, cigar):
#     """Get the reference locus, taking insertions/deletions into account"""
#     ref_pos = ref_start
#     seq_pos = 0
#
#     for (cigar_op, cigar_len) in cigar:
#         if cigar_op in (0, 7, 8):  # M, =, X: consume both ref and query
#             if seq_pos + cigar_len > seq_i:
#                 return ref_pos + (seq_i - seq_pos)
#             seq_pos += cigar_len
#             ref_pos += cigar_len
#         elif cigar_op == 1:  # I: consume query only
#             if seq_pos + cigar_len > seq_i:
#                 return None  # insertion before current position
#             seq_pos += cigar_len
#         elif cigar_op == 2:  # D: consume ref only
#             ref_pos += cigar_len
#         elif cigar_op == 3:  # N: skip region (introns typically)
#             ref_pos += cigar_len
#         elif cigar_op == 4:  # S: soft clipping, consume query only
#             if seq_pos + cigar_len > seq_i:
#                 return None  # soft clip before current position
#             seq_pos += cigar_len
#         elif cigar_op == 5:  # H: hard clipping, consume nothing
#             continue
#         elif cigar_op == 6:  # P: padding, consume neither
#             continue
#
#     return None  # position not reached, or it's past the alignment

class AlignmentFileFailure(Exception):
    pass

def count_alignment_error(max_errors:int|None=1000):
    global ALIGNMENT_ERROR_COUNT

    ALIGNMENT_ERROR_COUNT += 1
    if (max_errors is not None) and (ALIGNMENT_ERROR_COUNT > max_errors):
        raise AlignmentFileFailure(
            f"Too many errors detected in alignment files ({max_errors=}). "
            "Stopping processing on the assumption something is wrong and to avoid "
            "a 100+ GB error log."
        )


@define
class LocusValues:
    """Locus keyed values for an alignment.

    Attributes:
        methylations: dict mapping reference locus to methylation state (e.g. 'Z', 'z', 'X', etc.).
        qualities: dict mapping reference locus to PHRED quality score.
        nucleotides: dict mapping reference locus to nucleotide in the read.
    """
    methylations:dict[int, str]
    qualities:dict[int, int]
    nucleotides:dict[int, str]
    is_good:bool = True


@define(frozen=True)
class Alignment:
    """Wraps one or two pysam AlignedSegments, with methods for getting the
    methylation state at each reference locus.

    Merges overlapping paired alignments, using the mate with the highest quality score by default.

    In general, insertions and deletions are ignored.
    """
    a: AlignedSegment
    """AlignedSegment for read 1 (or the only read if single-ended)."""
    a2: AlignedSegment | None = None
    """AlignedSegment for read 2, or None if single-ended."""
    kind: Literal['bismark'] = 'bismark'
    filename: str = ''
    phred_offset: int = -33
    prefer_R1: bool = True
    """If True, locus_* properties will use read 1 for shared loci in paired alignments."""
    use_quality_profile:bool = False
    """ If True, locus_* properties will infer a PHRED score using both reads where they overlap.
            Based on Gaspar (2018)  https://doi.org/10.1186/s12859-018-2579-2. Overrides prefer_R1."""

    def _get_met_str(self, a:AlignedSegment):
        if self.kind == 'bismark':
            return get_bismark_met_str(a)
        else:
            raise NotImplementedError("Only bismark alignments are currently supported.")

    @property
    def R1(self) -> AlignedSegment:
        return self.a if self.a.is_read1 else self.a2
    @property
    def R2(self) -> AlignedSegment:
        return self.a2 if self.a.is_read1 else self.a

    def get_fwd_rev(self):
        """Returns the forward and reverse reads"""
        # if self.a.is_forward == self.a2.is_forward:
        #     logger.warning("Reads in same orientation, returned fwd/rev order is arbitrary.", )
        # actually that could cause a billion useless warnings depending on the sequencing technology
        fwd, rev = (self.a2, self.a) if self.a.is_reverse else (self.a, self.a2)
        return fwd, rev

    @property
    def segments(self) -> tuple[AlignedSegment, ...]:
        """Returns the two AlignedSegments in the Alignment."""
        return (self.a, self.a2) if self.a2 is not None else (self.a,)

    @property
    def fragment_is_plus(self) -> bool:
        """True if original fragment maps to the plus strand."""
        if self.R1.is_forward:
            return False
        else:
            return True

    @property
    def fragment_is_minus(self) -> bool:
        """True if original fragment maps to the minus strand."""
        return not self.fragment_is_plus

    def _has_metstr(self):
        """Checks both alignments for valid methylation strings"""
        if get_bismark_met_str(self.a) is None:
            return False
        if self.a2 is not None:
            if get_bismark_met_str(self.a2) is None:
                return False
        return True

    @cached_property
    def _a1_a2_overlaplen(self):
        """Returns alignments in position order and length of the overlap"""

        a1, a2 = self.a, self.a2

        # Put a1 on the left of a2
        if a1.reference_start > a2.reference_start:
            a2, a1 = a1, a2


        overlap_length = a1.reference_end - a2.reference_start
        return a1, a2, overlap_length

    @property
    def locus_methylation(self) -> dict[int, str]:
        """A dict mapping reference locus to methylation state (e.g. 'Z', 'z', 'X', etc.)."""
        return self._locus_values.methylations

    @property
    def locus_quality(self) -> dict[int, int]:
        """A dict mapping reference locus to PHRED quality score."""
        return self._locus_values.qualities

    @property
    def locus_nucleotide(self) -> dict[int, str]:
        """"A dict mapping reference locus to nucleotide in the read."""
        return self._locus_values.nucleotides


    @cached_property
    def _locus_values(self, ) \
            -> LocusValues:
        """Get object with attributes "qualities", "nucleotides", and "methylations"
        mapping each reference locus to the value, reference_locus -> value.
        """

        empty_return =  LocusValues({}, {}, {}, is_good=False)
        if not self._has_metstr():
            logger.warning(
                f"Can't determine metstr for alignment of {self.a.query_name}, skipping."
            )
            count_alignment_error()
            return empty_return

        for segment in (self.a, self.a2):
            if segment is None:
                continue

            align_len = sum([x[1] for x in segment.cigartuples if x[0] in {0, 1, 7, 8}])
            metstr_len = len(get_bismark_met_str(segment))
            phred_len = len(segment.query_qualities)
            nt_len = len(segment.query_sequence)
            if not (align_len == metstr_len == phred_len == nt_len):
                logger.warning(
                    f"Length mismatch of methylation string for alignment of {self.a.query_name}, skipping.\n"
                    f"({align_len=}, {metstr_len=}, {phred_len=}, {nt_len=} {segment.cigartuples=}, {self.filename=})"
                )
                count_alignment_error()
                return empty_return

        if self.a2 is None:
            phreds, methylations, nucleotides = {}, {}, {}
            for q_pos, r_pos in self.a.get_aligned_pairs(matches_only=True):
                phreds[r_pos] = self.a.query_qualities[q_pos]
                nucleotides[r_pos] = self.a.query_sequence[q_pos]
                methylations[r_pos] = self._get_met_str(self.a)[q_pos]
        else: # it's a paired alignment
            if self.use_quality_profile:
                from jtmethtools.quality_profiles import quality_profile_match_41, quality_profile_mismatch_41
            else:
                quality_profile_match_41, quality_profile_mismatch_41 = None, None

            # .aligned_pairs is reference position->read locus
            a1_loc_pos, a2_loc_pos = [
                {r: q for (q, r) in ap if (q is not None) and (r is not None)}
                for ap in (self.a.get_aligned_pairs(), self.a2.get_aligned_pairs())
            ]
            # a1_loc_pos, a2_loc_pos = [
            #     dict(aln.get_aligned_pairs(matches_only=True))
            #     for aln in (self.a, self.a2)
            # ]

            a1_loc = set(a1_loc_pos.keys())
            a2_loc = set(a2_loc_pos.keys())
            shared_loc = a1_loc.intersection(a2_loc)
            a1_only = a1_loc.difference(a2_loc)
            a2_only = a2_loc.difference(a1_loc)

            phreds = {}
            nucleotides = {}
            methylations = {}

            a1_metstr = get_bismark_met_str(self.a)
            a2_metstr = get_bismark_met_str(self.a2)

            try:
                # get the values where the position only exists in one of the mates
                for only_loc, loc_pos, a, metstr in (
                        (a1_only, a1_loc_pos, self.a, a1_metstr),
                        (a2_only, a2_loc_pos, self.a2, a2_metstr)
                ):
                    for l in only_loc:
                        p = loc_pos[l]
                        phreds[l] = a.query_qualities[p]
                        nucleotides[l] = a.query_sequence[p]
                        methylations[l] = metstr[p]
            except:
                print('BAM = ', self.filename, ' | Read = ', self.a.query_name)
                raise

            if (not self.prefer_R1) or self.use_quality_profile:
                for l in shared_loc:
                    a1_pos = a1_loc_pos[l]
                    a2_pos = a2_loc_pos[l]

                    a1_nt = self.a.query_sequence[a1_pos]
                    a2_nt = self.a2.query_sequence[a2_pos]

                    a1_phred = self.a.query_qualities[a1_pos]
                    a2_phred = self.a2.query_qualities[a2_pos]

                    a1_met = a1_metstr[a1_pos]
                    a2_met = a2_metstr[a2_pos]

                    # Keep the nucleotide with the highest quality
                    # (in the case that a1 and a2 have different nucleotides and
                    #   the phred is the same, we'll keep the a1 NT)
                    if a1_phred >= a2_phred:
                        nt = a1_nt
                        met = a1_met
                    else:
                        nt = a2_nt
                        met = a2_met

                    if self.use_quality_profile:
                        if a1_nt == a2_nt:
                            phred = quality_profile_match_41[a1_phred][a2_phred]
                        else:
                            phred = quality_profile_mismatch_41[a1_phred][a2_phred]
                    else:
                        phred = max((a1_phred, a2_phred))

                    phreds[l] = phred
                    nucleotides[l] = nt
                    methylations[l] = met
            else:
                # use R1
                for l in shared_loc:
                    a1, loc_pos, metstr = (
                        (self.a, a1_loc_pos, a1_metstr)
                        if self.a.is_read1
                        else (self.a2, a2_loc_pos, a2_metstr)
                    )
                    a1_pos = loc_pos[l]
                    phreds[l] = a1.query_qualities[a1_pos]
                    nucleotides[l] = a1.query_sequence[a1_pos]
                    methylations[l] = a1_metstr[a1_pos]

        sorted_phreds, sorted_nucleotides, sorted_methylations = (
            dict(sorted(d.items()))
            for d in (phreds, nucleotides, methylations)
        )

        return LocusValues(qualities=sorted_phreds, nucleotides=sorted_nucleotides, methylations=sorted_methylations)

    @property
    def fragment_length(self) -> typing.Union[int, Literal[np.nan]]:
        """Fragment length from the 5' ends of a read pair (cfDNAPro convention).

        Returns the distance from the 5' end of the forward-strand mate to the
        5' end of the reverse-strand mate. Unlike the outer span (or BAM TLEN),
        this stays correct for dovetailed pairs, where a fragment shorter than
        the read length causes each mate to run past the other's 5' end.

        Returns np.nan if the pair can't yield a meaningful length.
        """
        a, b = self.a, self.a2
        if b is None:
            return np.nan

        # Both mates must be mapped; reference_end is None for unmapped reads
        # and for reads whose CIGAR consumes no reference.
        if a.is_unmapped or b.is_unmapped:
            return np.nan
        if a.reference_end is None or b.reference_end is None:
            return np.nan

        # Same chromosome, opposite orientation (FR or RF).
        if a.is_reverse == b.is_reverse:
            return np.nan

        fwd, rev = self.get_fwd_rev()

        # reference_start is 0-based inclusive, reference_end 0-based exclusive,
        # so the difference is the span directly — no +1.
        length = rev.reference_end - fwd.reference_start

        # Outward-facing pairs give a negative or degenerate span.
        if length <= 0:
            return np.nan

        return length

    @property
    def fragment_start_end(self) -> tuple[int, int]:
        """Fragment start end position in reference positions, correcting for readthroughs.
        Where reads dovetail outwards, the overlapping span is used.
        See https://doi.org/10.1186/s13059-025-03607-5"""

        fwd, rev = self.get_fwd_rev()

        if fwd.reference_start > rev.reference_start:
            start = rev.reference_start
        else:
            start = min(fwd.reference_start, rev.reference_start)

        if rev.reference_end > fwd.reference_end:
            end = rev.reference_end
        else:
            end = max(fwd.reference_id, rev.reference_end)

        return start, end

    @property
    def methylation_values(self) -> list[str]:
        """Methylation values, in reference position order, without worrying about missing
        positions."""
        return list(self.locus_methylation.values())

    @property
    def metstr(self) -> str:
        """String of methylation states using the Bismark convention, merging both
         reads if paired, ordered by reference position. Gaps are represented by '-'.
         Non-reference positions are excluded.
         """

        metstr = []
        locs = list(self.locus_methylation.keys())
        locs = [x for x in locs if x is not None]

        # met = [x[1] for x in loc_met]

        start = min(locs)
        for pos in range(start, max(locs) + 1):
            m = self.locus_methylation.get(pos, None)
            if m is None:
                metstr.append('-')
            else:
                metstr.append(m)
        return ''.join(metstr)

    @property
    def reference_name(self) -> str:
        """Reference name (chromosome) of the alignment."""
        return self.a.reference_name

    @property
    def reference_start(self):
        """Start position of the alignment on the reference genome. If paired, returns
        the minimum start position of the two alignments."""
        if self.a2 is None:
            return self.a.reference_start
        else:
            return min(self.a.reference_start, self.a2.reference_start)

    @property
    def reference_end(self):
        """End position of the alignment on the reference genome. If paired, returns
        the maximum end position of the two alignments."""
        if self.a2 is None:
            return self.a.reference_end
        else:
            return max(self.a.reference_end, self.a2.reference_end)

    def mapping_quality(self):
        """Mapping quality of the alignment."""
        return self.a.mapping_quality


    def mCH(self) -> int:
        """Count number of methylated CH."""
        x = collections.Counter(self.methylation_values)
        return x['H'] + x['X'] + x['U']

    def _no_non_cpg(self) -> bool:
        """look for forbidden methylation states.

        Return True if there's no H|X|U."""
        values = set(self.methylation_values)
        if (
                ('H' in values)
                or ('X' in values)
                or ('U' in values)
        ):
            return False
        return True

    def has_methylated_ch(self) -> bool:
        """Return True if there is any methylated non-CpG in the alignment."""
        return not self._no_non_cpg()

    def get_hit_regions(self, regions: Regions) -> list[str]:
        """Get the names of the regions that this alignment overlaps.

        Returns an empty list if it doesn't overlap any region."""
        regions = [alignment_overlaps_region(a, regions)
                   for a in self.segments]
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
) -> Iterable[Tuple[AlignedSegment, Literal[None]]]:
    """Iterate over a single-ended bam file, yielding pairs of alignments.
    Where a read is unpaired, yield (alignment, None).

    Use start_stop for splitting a bam file for, e.g. multiprocessing.
    """

    bam = _load_bam(bam)

    for i, aln in enumerate(bam):
        logger.debug(f'Alignment #{i}')

        yield aln, None


def iter_bam_segments(
        bam: str | Path | AlignmentFile,
        paired_end: bool = True,
) -> Iterable[Tuple[AlignedSegment, AlignedSegment|None]]:
    """Iterate over a bam file, yielding pairs of alignments, or
    (segment, None) when it's unpaired.

    Supports SAM and BAM files.
    """
    #if check_pairedness, raises exception if bam is actually single-ended"""

    bam = _load_bam(bam)
    sorting_method = bam.header.get('HD', {}).get('SO', 'Unknown')
    logger.info(f'{sorting_method=}')

    if paired_end:
        if (sorting_method == 'queryname'):
            bamiter = _iter_bam_pe_qname_sorted(bam )
        elif (sorting_method == 'coordinate'):
            bamiter = _iter_pe_bam_unsorted(bam)
        else:
            raise RuntimeError(
                "Paired end BAM not sorted by queryname or coordinate. "
                "Sort it (with -n preferably, but by coordinate is fine), "
                "or use the --single-ended option.\n"
            )
    else:
        bamiter = _iter_bam_se(bam, )
    for aln in bamiter:
        yield aln
    return None


def iter_bam(
        bam:str|Path|AlignmentFile,
        paired_end:bool=True,
        kind='bismark',
        min_mapq:int=0,
        max_mCH:int=None,
) -> Iterable[Alignment]:
    """Iterate over a bam file, yielding Alignments."""

    for aln in iter_bam_segments(bam, paired_end,):
        aln = Alignment(*aln, kind=kind)
        if min_mapq and aln.mapping_quality() < min_mapq:
            continue
        if max_mCH is not None and aln.mCH() > max_mCH:
            continue
        yield aln
    return None
