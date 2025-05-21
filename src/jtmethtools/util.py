import typing
from pathlib import Path

import tarfile

import tempfile

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from typing import Tuple

import json
from loguru import logger
logger.remove()

from attrs import define

def set_logger(min_level='DEBUG'):

    logger.remove()
    if min_level in ('DEBUG', 'TRACE'):
        logger.add(print, level=min_level, filter=lambda record: record["level"].name in ('DEBUG', 'TRACE'))

    logger.add(lambda msg: print(f"\033[96m{msg}\033[0m"), level="INFO", format="{message}")


type SplitTable = dict[str, pd.DataFrame]

CANNONICAL_CHRM = [str(i) for i in range(1, 13)] + ['X', 'Y']
CANNONICAL_CHRM += ['chr' + c for c in CANNONICAL_CHRM]
CANNONICAL_CHRM = set(CANNONICAL_CHRM)


def fasta_to_dict(fn: str|Path, full_desc=False) -> dict[str, str]:
    """Dict that maps record_name->sequence.

    By default splits the description on the first space, this should
    give e.g. the chromosome name without extra metadata. Set full_desc
    to false to include whole description lines in the keys.
    """
    with open(fn) as f:

        gl = f.read().strip().split('\n')
    genome = {}
    chrm = None
    nt = None
    for line in gl:
        if line[0] == '>':
            if chrm is not None:
                genome[chrm] = ''.join(nt)
            chrm = line[1:]
            if not full_desc:
                chrm = line[1:].split()[0]
            nt = []
        else:
            nt.append(line.upper())
    return genome


def load_bismark_calls_table(fn) -> pd.DataFrame:
    df = pd.read_csv(fn, sep='\t', header=None, dtype={2: str})
    df.columns = ['ReadName', 'Methylated', 'Chromosome', 'Locus', 'Call']
    return df


def split_table_by_chrm(table:pd.DataFrame, chrm_col='Chrm') \
        -> SplitTable:
    """Split a table by chromosomes, returning dict keyed by each
    chromosome. New split tables indexed by Locus."""

    chrm_table = {
        c: table.loc[table[chrm_col] == c]
        for c in table[chrm_col].unique()
    }
    if "Locus" in table:
        chrm_table = {
            c:t.set_index('Locus', drop=False)
            for c, t in chrm_table.items()
        }

    return chrm_table


def load_region_bed(fn) -> pd.DataFrame:
    regions = pd.read_csv(
        fn, sep='\t', header=None,
        dtype={0:str}
    )
    regions.columns = ['Chrm', 'Start', 'End', 'Name', ]

    regions.set_index('Name', inplace=True, drop=False)
    return regions


def write_array(
        array: NDArray,
        outfile: str|Path|typing.IO,
        additional_metadata: dict = None,
        gzip:bool='infer',
) -> None:
    """Write a tar file that contains the binary data and metadata
    to recreate the original numpy array.

    additional_metadata values should be strings or stringable."""
    # Create a temporary directory to store the files

    if isinstance(outfile, str) or isinstance(outfile, Path):
        tararg = {'name':outfile}
    else:
        tararg = {'fileobj':outfile}

    if gzip == 'infer':
        gzip = True if str(outfile).endswith('.gz') else False
    mode = 'w:gz' if gzip else 'w'

    with tarfile.open(**tararg, mode=mode) as tar:
        # Save the binary data of the array
        with tempfile.NamedTemporaryFile('w') as tmpf:

            array.tofile(tmpf.name)
            tar.add(tmpf.name,  arcname='data.bin')

        # Create the metadata file with shape and dtype
        with tempfile.NamedTemporaryFile('w') as tmpf:

            metadata = {
                f'_np_shape': [s for s in array.shape],
                f'_np_dtype': str(array.dtype),
            }

            if additional_metadata is not None:
                metadata = metadata | additional_metadata

            json.dump(metadata, tmpf)
            tmpf.flush() # write the buffer to disk

            tar.add(tmpf.name, arcname='metadata.json')



def read_array(file: str|Path|typing.IO,
               gzip='infer') -> Tuple[NDArray, dict]:
    """Read file created by `write_array`"""
    if isinstance(file, str) or isinstance(file, Path):
        tararg = {'name':file}
    else:
        tararg = {'fileobj':file}

    if gzip == 'infer':
        if str(file).endswith('.gz'):
            gzip = True
        else:
            gzip = False

    mode = 'r:gz' if gzip else 'r'

    with tarfile.open(**tararg, mode=mode) as tar:

        # Extract the metadata file and parse it
        meta_file = tar.extractfile("metadata.json").read().decode('utf-8')

        metadata = json.loads(meta_file)

        # Extract the binary data and load it as a NumPy array
        data_file = tar.extractfile("data.bin")

        array = np.frombuffer(
            data_file.read(),
            dtype=metadata['_np_dtype']
        ).reshape(metadata['_np_shape'])

    return array, metadata

@define(slots=False)
class MockAlignment:
    query_name: str = 'test'
    reference_name: str = '1'
    reference_start: int = 100
    _reference_end: int = None
    mapping_quality: int = 44
    _cigartuples: list[tuple[int, int]] = None
    _query_qualities: list[int] = None
    _query_sequence: str = None
    _aligned_pairs: list[tuple[int, int]] = None
    meth_str: str = 'x.ZzZ.u.h.'

    should_succeed: bool = True
    expected_score: float = np.nan

    def get_tags(self):
        return (None, None, ('XM', self.meth_str))

    @property
    def cigartuples(self):
        if self._cigartuples is None:
            return  [(0, len(self.meth_str))]
        else:
            return self._cigartuples

    def get_aligned_pairs(self, matches_only=True):
        if self._aligned_pairs is not None:
            return self._aligned_pairs
        if len(self.cigartuples) != 1:
            raise NotImplementedError("MockAlignment can't deal with indels")
        return [(i,locus) for i, locus in enumerate(range(self.reference_start, self.reference_end))]

    @property
    def query_qualities(self):
        if self._query_qualities is None:
            return [41 for _ in self.meth_str]
        else:
            return self._query_qualities

    @property
    def query_sequence(self):
        if self._query_sequence is None:
            return ''.join(['N' for _ in self.meth_str])
        else:
            return self._query_sequence

    @property
    def reference_end(self):
        if self._reference_end is None:
            return self.reference_start + len(self.meth_str)
        else:
            return self._reference_end


import random
import string

nt_meth_mapper = dict()
nts = 'ACTG'
for nt1 in nts:
    for nt2 in nts:
        nt_meth_mapper[nt1 + nt2] = '.'
for nt in nts:
    nt_meth_mapper['C' + nt] = 'H'
for nt in nts:
    nt_meth_mapper['T' + nt] = 'h'

nt_meth_mapper['CG'] = 'Z'
nt_meth_mapper['TG'] = 'z'
del nt, nts

class SamAlignment:
    def __init__(
            self,
            seq: str = None, qname: str = None, flag: int = 0, rname: str = "1",
            pos: int = 0, mapq: int = 60, cigar: str = None, rnext: str = "*",
            pnext: int = 0, tlen: int = 10, qual: str = None,
            methylation: str = None,
    ):
        """Some default values calculated to make sense with given sequence or tlen.
        If only seq given then methylation string determined, and vis versa.

        Caveats:
            - Doesn't deal with indels in a sensible way, you'll need to check that
              yourself.
            - Generated seq or methylation assumes all methylation is on the forward
              strand.

        """

        # If a sequence is given, calculate the template length (TLEN)
        self.tlen = tlen or len(seq)

        # add CpG sites from CGs in seq, if no methylation given
        if methylation is None:
            if seq:
                methylation = ''.join([
                    nt_meth_mapper[seq[i:i + 2]]
                    for i in range(len(seq) - 1)
                ])
            else:
                methylation = '.' * self.tlen
        self.methylation = methylation

        # set seq based on the methylation string, if missing
        if seq is None:
            s = []
            i = 0
            while i < len(methylation):
                if methylation[i] == 'Z':
                    s.append('CG')
                elif methylation[i] == 'z':
                    s.append('TG')
                elif methylation[i] == 'h':
                    s.append('TA')
                elif methylation[i] == 'H':
                    s.append('CA')
                else:
                    s.append('A')
                i = len(s)

            # Default values if the arguments are not provided
        self.qname = qname or ''.join(random.choices(string.ascii_letters, k=10))
        self.seq = seq or 'N' * self.tlen
        self.cigar = cigar or str(len(self.seq)) + 'M'
        self.qual = qual or "J" * len(self.seq)

        # Other fields
        self.flag = flag
        self.rname = rname
        self.pos = pos
        self.mapq = mapq
        self.rnext = rnext
        self.pnext = pnext

    def __str__(self):
        # Create SAM record as a string
        sam_record = '\t'.join([
            str(x) for x in [
            self.qname,
            self.flag,
            self.rname,
            self.pos,
            self.mapq,
            self.cigar,
            self.rnext,
            self.pnext,
            self.tlen,
            self.seq,
            self.qual,
            f"NM:i:tag0\tMD:Z:tag1\tXM:Z:{self.methylation}"
        ]])

        return sam_record


def generate_sam_file(alignments: list[SamAlignment], sorting: str = 'coordinate') -> str:
    """Output a valid SAM file for given list of alignments. Alignments will be
    sorted according to `sorting`. Valid options: coordinate, queryname, unsorted
    and unknown"""
    # Validate the sorting option
    valid_sortings = {'coordinate', 'queryname', 'unsorted', 'unknown'}
    if sorting not in valid_sortings:
        raise ValueError(
            f"Invalid sorting option '{sorting}'. "
            f"Valid options are {valid_sortings}")

    # Sort the alignments if required
    if sorting == 'coordinate':
        alignments.sort(key=lambda x: x.pos)
    elif sorting == 'queryname':
        alignments.sort(key=lambda x: x.qname)

    # Collect all unique reference names (rname) that are not '*'
    rnames = set(alignment.rname for alignment in alignments if alignment.rname != "*")

    # Create header, @SQ for each unique reference name
    header_lines = [f"@HD	VN:1.0	SO:{sorting}"]
    for rname in rnames:
        # Determine the maximum position in the alignments for this reference name
        max_pos = max(alignment.pos + len(alignment.seq) for alignment in alignments if alignment.rname == rname)
        header_lines.append(f"@SQ\tSN:{rname}\tLN:{max_pos}")

    # Combine the header and the alignment records
    sam_file = "\n".join(header_lines) + "\n"
    sam_file += "\n".join(str(alignment) for alignment in alignments)

    return sam_file