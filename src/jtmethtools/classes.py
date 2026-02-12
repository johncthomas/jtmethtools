import typing
from functools import cached_property, lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TypedDict, Self, Union

import numpy as np
import pandas as pd
from attrs import define
from jtmethtools.util import (
    load_region_bed, split_table_by_chrm,
    fasta_to_dict, CANNONICAL_CHRM
)
import pickle
from numpy import typing as npt
from numpy.typing import NDArray
import re

type Pathy = str|Path

__all__ = ['Regions', 'CpGIndex', 'Genome',  'LociRange']


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
    df: pd.DataFrame = None

    def __attrs_post_init__(self):
        # make it insensitive to presence/absence of 'chr' in chromosome names
        chromosomes =  list(self.starts.keys())
        for c in chromosomes:
            if c.startswith('chr'):
                nuchrm = c[3:]
            else:
                nuchrm = 'chr' + c

            self.starts[nuchrm] = self.starts[c]
            self.ends[nuchrm] = self.ends[c]
            self.names[nuchrm] = self.names[c]

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
        return set(self.starts.keys())

    @classmethod
    def from_file(cls,  filename: Pathy, sep='\t',) -> Self:
        """Expects either a BED file or a TSV with the columns Chrm, Start, End, Name.

        Set `sep` to specify a separator for the TSV.

        If it's a BED file that doesn't end with .bed or .bed.gz, use Regions.from_bed."""
        filename = str(filename)
        if filename.endswith('.bed') or filename.endswith('.bed.gz'):
            return cls.from_bed(filename)

        df = pd.read_csv(filename, sep=sep, dtype={'Chrm':str})
        return cls.from_df(df)


    @classmethod
    def from_bed(cls, filename: Pathy) -> Self:
        df = load_region_bed(filename)
        return cls.from_df(df)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """Expects columns Start, End, Name & Chrm."""
        df.set_index('Name', inplace=True, drop=False)
        # safely check Chrm is str
        if not pd.api.types.is_string_dtype(df.Chrm.dtype):
            chrms = df.Chrm.astype(str)
            # pandas 3 might not allow changing dtypes
            df = df.drop('Chrm', axis=1)
            df.loc[:, 'Chrm'] = chrms
        sdf = split_table_by_chrm(df)


        return cls(
            starts={k: sdf[k].Start.values for k in sdf},
            ends={k: sdf[k].End.values for k in sdf},
            names={k: sdf[k].Name.values for k in sdf},
            df=df
        )

    def starts_ends_of_chrm(self, chrm) -> (NDArray[int], NDArray[int]):
        return (self.starts[chrm], self.ends[chrm])

    def region_at_locus(self, chrm:str, locus:int, missing_value=False) \
            -> Union[str,False]:
        """Return region name at locus. Does not check for overlapping regions.
        Return `missing_value` if no region hit."""
        if chrm not in self.chromsomes:
            return missing_value
        m = (locus >= self.starts[chrm]) & (locus < self.ends[chrm])
        if any(m):
            return self.names[chrm][m][0]
        else:
            return missing_value


@define(frozen=True)
class CpGIndex:
    """Index of CpG sites in a genome.

    Attributes:
        cpg_list: a tuple of (chrm, locus) tuples
        locus2index: maps (chrm, locus) -> cpg_index
        region_names: optional tuple of region names for each CpG site

    Note: Uses MappingProxyType for locus2index to make it immutable. If you wish to
    modify, make changes to cpg_list and then create a new instance using
    CpGIndex.from_cpg_list.
    """
    cpg_list: tuple[tuple[str, int], ...]
    locus2index: MappingProxyType[tuple[str, int], int]
    region_names: tuple[str, ...] = None

    @classmethod
    def from_cpg_list(
            cls,
            cpg_list: list[tuple[str, int]],
            region_names: tuple[str, ...] = None
    ) -> Self:
        """Create a CpGIndex from a list of (chromosome, position) tuples."""
        pos2index = {
            pos: idx for idx, pos in enumerate(cpg_list)
        }
        pos2index = pos2index | {
            (c, p+1): idx for idx, (c, p) in enumerate(cpg_list)
        }
        pos2index = MappingProxyType(pos2index)
        return cls(cpg_list=tuple(cpg_list), locus2index=pos2index, region_names=region_names)

    @classmethod
    def from_genome(cls, genome: 'Genome', region_names: tuple[str] = None) -> Self:
        """Create a CpGIndex from a Genome object."""
        cpg_list = []
        for chrm, seq in genome.sequences.items():
            cpg_list.extend((chrm, m.start()) for m in re.finditer('(?=CG)', seq))
        return cls.from_cpg_list(cpg_list, region_names=region_names)

    @classmethod
    def from_fasta(cls, fn: str | Path, region_names: tuple[str] = None) -> Self:
        """Create a CpGIndex from a fasta file."""
        genome = Genome.from_fasta(fn)
        return cls.from_genome(genome, region_names=region_names)

    def filter_cpg_index_by_regions(
            self,
            regions: Regions,
    ) -> Self:

        new_cpg_list = []
        cpg_regions:list[str] = []
        for chrm, loc in self.cpg_list:

            hit = regions.region_at_locus(chrm, loc)
            if hit:
                cpg_regions.append(hit)
                new_cpg_list.append((chrm, loc))

        return CpGIndex.from_cpg_list(new_cpg_list, region_names=tuple(cpg_regions))

    def __hash__(self):
        # as it's frozen, we can use the id of the object as a hash.
        return id(self)

    def to_file(self, fn: str | Path) -> None:
        """Write the CpG index to a file (pickle)."""
        with open(fn, 'wb') as f:
            dat = {'cpg_list': self.cpg_list,
                   'locus2index': dict(self.locus2index),
                   'region_names': self.region_names}
            pickle.dump(dat, f)

    @classmethod
    def from_file(cls, fn: str | Path) -> Self:
        """Load a CpG index from a file (pickle)."""
        with open(fn, 'rb') as f:
            dat = pickle.load(f)
            dat['locus2index'] = MappingProxyType(dat['locus2index'])
            return cls(
                **dat,
            )

@define(frozen=True)
class Genome:
    """Class for holding and working with dictionary of sequences."""
    sequences: MappingProxyType[str, str]
    filename: str | Path = None

    def harmonise_chrm_names(self) -> Self:
        """add chr to chromosome names that don't have it, and remove it from them that do.

        Creates a new instance of Genome as it's supposed to be immutable."""
        new_seqs = {k: v for k, v in self.sequences.items()}
        for k in self.sequences.keys():
            if k.startswith('chr'):
                new_seqs[k[3:]] = self.sequences[k]
            else:
                new_seqs['chr' + k] = self.sequences[k]

        return Genome.from_dict(sequences=new_seqs, filename=self.filename)

    @classmethod
    def from_fasta(cls, fn: str | Path,
                   full_desc=False, cannonical_only=True) -> Self:
        """Load a fasta file into a Genome object"""
        seqs =  fasta_to_dict(fn, full_desc=full_desc)
        if cannonical_only:
            seqs = MappingProxyType({
                k: v for k, v in seqs.items() if k in CANNONICAL_CHRM
            })
        genome = cls(
            MappingProxyType(
                seqs
            ),
            filename=fn
        )
        return genome

    @classmethod
    def from_dict(cls, sequences: dict[str, str],
                  filename='<from dict>', cannonical_only=True,) -> Self:
        """Create a Genome object from a dict.

        Interally uses MappingProxyType for immutable dict like."""
        if cannonical_only:
            sequences = {
                k: v for k, v in sequences.items() if k in CANNONICAL_CHRM
            }
        return cls(MappingProxyType(sequences), filename=filename)

    def __getitem__(self, key):
        return self.sequences[key]

    @cached_property
    def cpg_index(self) -> CpGIndex:
        idx = CpGIndex.from_genome(self)
        return idx

    @property
    def chromsomes(self) -> list[str]:
        return list(self.sequences.keys())

    def __repr__(self):
        return f"<Genome: filename={self.filename},  id={id(self)}>"

    def __hash__(self):
        # attrs version of the hash tries to hash .sequences,
        #   which raises an error.
        return id(self)





def ttest_genome_cpg_index():
    testgen = Genome.from_dict(
             # 0123456789012
             # ___x_x___x___
        {'a': 'TTTCGCGTTCGGC'},
        cannonical_only=False
    )
    #print(testgen.sequences)
    test_cindex = CpGIndex.from_genome(testgen)
    #print(test_cindex.cpg_list)
    assert test_cindex.cpg_list == (('a', 3), ('a', 5), ('a', 9))
    assert test_cindex.locus2index == MappingProxyType({('a', 3): 0, ('a', 5): 1, ('a', 9): 2})


    regions_df = pd.DataFrame({
        'Chrm': ['a'],
        'Start': [4],
        'End': [6],
        'Name': ['test_region']
    })
    r = Regions.from_df(regions_df)
    t2 = test_cindex.filter_cpg_index_by_regions(r)
    assert t2.cpg_list == ( ('a', 5), )
    assert t2.region_names == ('test_region',)

if __name__ == '__main__':
    ttest_genome_cpg_index()
    print("All tests passed.")