import typing
from pathlib import Path
import random, string
import tarfile

import tempfile


import pyarrow as pa
import pandas as pd
import numpy as np
from numpy.typing import NDArray

from typing import Tuple, Mapping
from datetime import datetime
import json
from loguru import logger


logger.remove()

from attrs import define


"""Utility functions, including some for working with Bismark tables."""


__all__ = ["CANNONICAL_CHRM", "setup_logfile", "fasta_to_dict", "read_bismark_calls_table",
           "split_table_by_chrm", "read_region_bed", "read_cov", "write_cov", "filter_cov",
           "write_array", "read_array", "read_bed", "table2df", "read_parquet", "log_memory_footprint"]

def set_logger(min_level='DEBUG'):

    logger.remove()
    if min_level in ('DEBUG', 'TRACE'):
        logger.add(print, level=min_level, filter=lambda record: record["level"].name in ('DEBUG', 'TRACE'))

    logger.add(lambda msg: print(f"\033[96m{msg}\033[0m"), level="INFO", format="{message}")


SplitTable = dict[str, pd.DataFrame]

CANNONICAL_CHRM = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
CANNONICAL_CHRM += ['chr' + c for c in CANNONICAL_CHRM]
CANNONICAL_CHRM = set(CANNONICAL_CHRM)
"""Numbered chromosomes, plus X, Y and MT, with and without 'chr' prefix. Used for filtering out non-standard contigs."""

def setup_logfile(log_dir: Path, sample: str) -> int:
    """Set up a log file in with path {log_dir}/{sample}.{timestamp}.log,
    and add it as a sink to the logger.

    Returns the log ID of the added sink, which can be used to remove it later if needed."""
    now = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{sample}.{now}.log"
    log_id = logger.add(log_path,
                        level='INFO')
    return log_id


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
    genome[chrm] = ''.join(nt)
    return genome


def read_bismark_calls_table(fn) -> pd.DataFrame:
    """Load a Bismark methylation calls file into a DataFrame with columns
    ReadName, Methylated, Chromosome, Locus, Call."""
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


def read_region_bed(fn) -> pd.DataFrame:
    """Load a BED file with regions.

    Return a DataFrame indexed by region name. If present (and made
    of unique values), the 4th column is used as the region name.
    Otherwise region name is constructed in the Chrm:Start-End
    format (zero-based).

    Other columns are dropped."""
    regions = pd.read_csv(
        fn, sep='\t', header=None,
        dtype={0: str}
    )

    # deal with the fact most of our bed files have names in the 4th column
    #  but the contents after the 3rd column are not always the same.
    if (regions.shape[1] > 3) and regions[3].is_unique:

        # if the 4th column contains unique values, use it as names
        regions = regions.iloc[:, :4]
    else:
        regions = regions.iloc[:, :3]
        # otherwise use the first 3 columns as the region name
        regions['Name'] = regions[0].astype(str) + ':' + \
                          regions[1].astype(str) + '-' + regions[2].astype(str)

    regions.columns = ['Chrm', 'Start', 'End', 'Name']
    regions.set_index('Name', inplace=True, drop=False)
    return regions


def read_cov(fn, locus2index:Mapping[tuple[str, int], int]=None):
    """Load a Bismark coverage file into a DataFrame with columns Chrm, Position,
    Perc, M, U, Depth, Beta.

    If locus2index maps (chrm, position) -> CpG number. If provided, will sum
    strands together and return a DF indexed by CpG index. Otherwise, returns
    the cov file as a DF indexed by (chrm, start).

    Remember Bismark are 1 indexed.
    """
    df = pd.read_csv(fn, sep='\t', header=None, dtype={0:str})
    df.columns = ['Chrm', 'Position', 'End', 'Perc', 'M', 'U']
    df.drop('End', axis=1, inplace=True)
    df.loc[:, 'Depth'] = df.M + df.U
    df.loc[:, 'Beta'] = df.M / df.Depth
    df.set_index(['Chrm', 'Position'], drop=False, inplace=True)
    # Change the names to avoid annoying "ambiguous column name" errors
    df.index.names = ['ChrmIdx', 'StartIdx']
    if locus2index is not None:
        df = sum_strands(df, locus2index)
    return df


def sum_strands(cov: pd.DataFrame, locus2index: dict[tuple[str, int], int]):
    """Sum counts for both positions in a CpG.

    Args:
        cov: DataFrame indexed by (chrm, position) with columns M & U.
        locus2index: Dictionary mapping (chrm, position) to CpG indicies used to sum
    the sites together.

    If the number of sites found is low, remember that bismark cov is 1-indexed."""
    cov = cov.copy()
    idx = cov.index.map(lambda x: locus2index.get(x, -1))
    cov.loc[:, 'CpGIndex'] = idx.values
    cov = cov.loc[idx != -1]
    cov = cov.groupby('CpGIndex')[['M', 'U']].sum()
    cov.loc[:, 'Depth'] = cov.M + cov.U
    cov.loc[:, 'Beta'] = cov.M / cov.Depth
    return cov


def write_cov(df, fn):
    """Write a cov table from DF with columns Chrm, Position, Perc, M, U.

    Drops other columns if they are present."""
    req_cols = ['Chrm', 'Position', 'Perc', 'M', 'U']
    if not all([c in df.columns for c in req_cols]):
        raise ValueError(
            f"DataFrame must contain columns {req_cols} write cov file.\n"
            f"Columns found: {df.columns.tolist()}"
        )

    # Deliberately writing Position twice, in a cov BED table the Start position is
    #  always the same as the End position.
    df \
        .reindex(columns=['Chrm', 'Position', 'Position', 'Perc', 'M', 'U']) \
        .to_csv(fn, sep='\t', header=False, index=False)


def filter_cov(cov_df, included_sites: pd.Index, min_depth=50) -> pd.DataFrame:
    """Filter coverage table by loci present in the included_sites, and depth.
    Set min_depth=None to avoid that filter"""
    cov_filt = cov_df.reindex(included_sites).dropna()
    if min_depth is not None:
        cov_filt = cov_filt.loc[cov_filt.Depth >= min_depth]
    return cov_filt


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


def bismark_methylation_calls(
    bs_seq: str,
    genome_seq: str
) -> str:
    """
    Generate a Bismark-style methylation call string for a bisulfite read.
    Forward read is assumed.

    Parameters
    ----------
    bs_seq
        Bisulfite‐converted read sequence (A/C/G/T), same length as genome_seq.
    genome_seq
        Reference genome sequence aligned to the read, same length as bs_seq.

    Returns
    -------
    str
        Methylation call string, same length as inputs, with:
          Z/z for CpG,
          X/x for CHG,
          H/h for CHH,
          . elsewhere.
    """
    calls: list[str] = []
    L = len(genome_seq)

    # upper‐case everything so we can ignore case
    bs = bs_seq.upper()
    ref = genome_seq.upper()

    for i, (g, r) in enumerate(zip(ref, bs)):
        if g == 'C':
            # determine context
            if i + 1 < L and ref[i+1] == 'G':
                up, lo = 'Z', 'z'
            elif i + 2 < L and ref[i+2] == 'G':
                up, lo = 'X', 'x'
            else:
                up, lo = 'H', 'h'
            # methylation call
            if r == 'C':
                calls.append(up)
            elif r == 'T':
                calls.append(lo)
            else:
                calls.append('.')
        else:
            calls.append('.')

    return ''.join(calls)


class SamAlignment:
    def __init__(
            self,
            seq: str = None, qname: str = None, flag: int = 2, rname: str = "1",
            pos: int = 1, mapq: int = 60, cigar: str = None, rnext: str = "*",
            pnext: int = 0, tlen: int = 10, qual: str = None,
            methylation: str = None, genome: str = None,
    ):
        """Some default values calculated to make sense with given parameters.
        If seq and genome provided, a methylation string is generated. If seq is None,
        but a methylation string is provided, a valid nulceotide sequence is generated.

        Unless passed with a valid nucleotide string, methylation has to be even in length
        with cytosines, i.e. "Z.z...h."

        Otherwise default methylation is '.'*tlen and seq is 'N'*tlen.

        Caveats:
            - Doesn't deal with indels in a sensible way, you'll need to provide full
              prameters.
            - Generated seq or methylation assumes all methylation is on the forward
              strand.
        """

        if seq and methylation:
            assert len(seq) == len(methylation)
        if seq:
            tlen = len(seq)
        if methylation:
            tlen = len(methylation)

        # add CpG sites from CGs in seq, if no methylation given
        if methylation is None:
            if seq and genome:
                methylation = bismark_methylation_calls(seq, genome)
            else:
                methylation = '.' * tlen
        else:
            if len(methylation) % 2:
                raise ValueError('methylation has to be an even number in length')
        self.methylation = methylation

        # set seq based on the methylation string, if missing
        if (seq is None) and (methylation is not None):
            s = ''
            i = 0
            while i < len(methylation):
                if methylation[i] == 'Z':
                    s += 'CG'
                elif methylation[i] == 'z':
                    s += 'TG'
                elif methylation[i] == 'h':
                    s += 'TA'
                elif methylation[i] == 'H':
                    s += 'CA'
                else:
                    s += 'A'
                i = len(s)
            seq = ''.join(s)

        # if seq is still None, make sure it's set
        if seq is None:
            seq = 'N' * tlen

        self.tlen = tlen

        # Default values if the arguments are not provided
        self.qname = qname or ''.join(random.choices(string.ascii_letters, k=10))
        self.seq = seq
        self.cigar = cigar or str(tlen) + 'M'
        self.qual = qual or "J" * tlen

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


# Create mappings for dictionary encodings
def mapping_to_pa_dict(mapping:dict[str, int], values:NDArray, ) -> pa.DictionaryArray:

    """Take a encoded numpy array and the dictionary that decodes
    it, return the dictionary encoded arrow array."""
    # get values in the order of the mapping so they'll be the same in the
    #  final dictionary
    i2s = {v:k for k, v in mapping.items()}
    dict_values = [i2s[i] for i in range(max(i2s.keys()) + 1)]
    dictionary = pa.array(dict_values, type=pa.string())

    # Convert to dict array. Pandas needs 32bit array to convert to
    indices = pa.array(values, type=pa.int32())
    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    return dict_array


def index_of_true(bool_series: pd.Series) -> pd.Index:
    """Return the index of a boolean series where the value is True."""
    return bool_series.loc[bool_series].index


def read_bed(fn):
    df:pd.DataFrame = pd.read_csv(fn, sep='\t', header=None, dtype={0:str})
    df.rename(columns={0:'Chrm', 1:'Position', 2:'End'}, inplace=True)
    return df


def table2df(table: pa.Table) -> pd.DataFrame:
    """Convert a pyarrow Table to a pandas DataFrame, converting dictionary types to pandas categorical types."""
    mapping = {schema.type: pd.ArrowDtype(schema.type) for schema in table.schema}

    return table.to_pandas(types_mapper=mapping.get, ignore_metadata=True)


def read_parquet(fn) -> pd.DataFrame:
    """load a parquet file and convert to a pandas dataframe.
    Works when pd.read_parquet fails on the dictionary types."""
    tls = pa.parquet.read_table(fn)
    tls = table2df(tls)
    return tls


def log_memory_footprint():
    import psutil
    import os

    # Get the current process
    process = psutil.Process(os.getpid())

    # Get memory usage in bytes
    memory_usage = process.memory_info().rss  # in bytes

    # Convert to megabytes (MB) for easier readability
    memory_usage_mb = memory_usage / (1024 ** 2)

    logger.info(f"Memory usage: {memory_usage_mb:.2f} MB")
