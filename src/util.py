import typing
from typing import Collection, Tuple

import pandas as pd
import numpy as np
import pysam
from loguru import logger
from pysam import AlignedSegment, AlignmentFile, AlignmentHeader
from attrs import define, field
from functools import cached_property


SplitTable = dict[str, pd.DataFrame]


def fasta_to_dict(fn: str, full_desc=False) -> dict[str, str]:
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
            if full_desc:
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
    chromosome."""
    return {c: table.loc[table[chrm_col] == c] for c in table[chrm_col].unique()}


def load_region_bed(fn):
    regions = pd.read_csv(
        fn, sep='\t', header=None,
        dtype={'Chrm':str}
    )
    regions.columns = ['Chrm', 'Start', 'End', 'Name', ]

    regions.set_index('Name', inplace=True, drop=False)
    return regions
