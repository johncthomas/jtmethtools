import typing
from typing import Collection, Tuple, Self

import pandas as pd
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
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

#
# @define
# class Regions:
#     """Region starts, ends and names stored in vectors
#     as attributes of the same names.
#
#     Create using Regions.from_file or .from_df
#
#     Methods:
#         starts_ends_of_chrm: get start and end vectors for a chromosome
#     """
#     starts: dict[str, NDArray[int]]
#     ends: dict[str, NDArray[int]]
#     names: dict[str, NDArray[str]]
#     thresholds: dict[str, float] = None
#     df: pd.DataFrame = None
#
#     @cached_property
#     def chromsomes(self) -> set[str]:
#         return set(self.df.Chrm.unique())
#
#     @classmethod
#     def from_table(cls, filename: str) -> Self:
#         df = pd.read_csv(filename, sep='\t')
#         return (cls.from_df(df))
#
#
#     @classmethod
#     def from_bed(cls, filename: str) -> Self:
#         df = load_region_bed(filename)
#         return cls.from_df(df)
#
#     @classmethod
#     def from_df(cls, df: pd.DataFrame) -> Self:
#         sdf = split_table_by_chrm(df)
#
#         return cls(
#             starts={k: sdf[k].Start.values for k in sdf},
#             ends={k: sdf[k].End.values for k in sdf},
#             names={k: sdf[k].Name.values for k in sdf},
#             thresholds=df.Threshold.to_dict() if 'Threshold' in df.columns.values else None,
#             df=df
#         )
#
#     def starts_ends_of_chrm(self, chrm) -> (NDArray[int], NDArray[int]):
#         return (self.starts[chrm], self.ends[chrm])
#
#     def get_region_threshold(self, name):
#         return self.thresholds[name]