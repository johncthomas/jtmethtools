"""Classes for working with genomic sequences, regions and CpG sites. Available in the package main namespace."""

import typing
from functools import cached_property, lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TypedDict, Self, Union

import numpy as np
import pandas as pd
from attrs import define
from jtmethtools.util import (
    read_region_bed, split_table_by_chrm,
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
    """A single genomic region, with start, end, chromosome and optional name."""
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
    """Region starts, ends and names stored in vectors.

    Primarily passed to filter functions of other classes.

    Is indifferent about chromosome names containing "chr" or not.
    """
    starts: dict[str, NDArray[int]]
    """(chrm -> vector of region starts)"""
    ends: dict[str, NDArray[int]]
    """(chrm -> vector of region ends)"""
    names: dict[str, NDArray[str]]
    """(chrm -> vector of region names)"""
    df: pd.DataFrame = None
    """Above in the form of a dataframe. For reference only, changes to the dataframe will not be reflected 
    in the starts/ends/names dicts."""

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
        """Will attempt to use any 4th column as region names. Absent or non-unique 4th column
        will result in names in format "Chrm:Start-End"."""
        df = read_region_bed(filename)
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
        cpg_list: a tuple of (chrm, locus) tuples.
        locus2index: maps (chrm, locus) -> cpg_index
        region_names: optional tuple of region names for each CpG site
        one_indexed: True if locus2index is uses 1-indexing for chromosome position.


    Note: Uses MappingProxyType for locus2index to make it immutable. If you wish to
    modify, make changes to cpg_list and then create a new instance using
    CpGIndex.from_cpg_list.
    """
    # NOTE: all constructors should call from_cpg_list.

    # note to copilot: there are two positions in each CpG, the second +1 base after the first.
    # This is always true whether the 2 given positions are 1-indexed or not.

    cpg_list: tuple[tuple[str, int], ...]
    locus2index: MappingProxyType[tuple[str, int], int]
    one_indexed: bool = False
    region_names: tuple[str, ...] = None

    @classmethod
    def from_cpg_list(
            cls,
            cpg_list: list[tuple[str, int]],
            is_one_indexed:bool,
            make_one_indexed: bool = False,
            region_names: tuple[str, ...] = None,

    ) -> Self:
        """Create a CpGIndex from a list of (chromosome, position) tuples.

        If CpG list position is already one indexed, set `is_one_indexed` to True.
        If you want to make it one indexed, set `make_one_indexed` to True, this will set
            the one_indexed attribute to True and add 1 to all positions in the cpg_list and locus2index."""
        if is_one_indexed and make_one_indexed:
            raise ValueError("Cannot set both is_one_indexed and make_one_indexed to True. "
                             "If it's already one indexed, there's no need to make it one indexed.")
        pos2index = {
            (pos[0], pos[1]): idx for idx, pos in enumerate(cpg_list)
        }
        # create mapping for second C in CpG.
        pos2index |= {
            (c, p+1): idx for idx, (c, p) in enumerate(cpg_list)
        }

        pos2index = MappingProxyType(pos2index)
        return cls(cpg_list=tuple(cpg_list), locus2index=pos2index,
                   region_names=region_names, one_indexed=make_one_indexed or is_one_indexed)

    @classmethod
    def from_genome(
            cls,
            genome: 'Genome',
            region_names: tuple[str] = None,
            make_one_indexed: bool = False
    ) -> Self:
        """Create a CpGIndex from a Genome object."""
        cpg_list = []
        offset = 1 if make_one_indexed else 0
        for chrm, seq in genome.sequences.items():
            cpg_list.extend((chrm, m.start()+offset) for m in re.finditer('(?=CG)', seq))
        return cls.from_cpg_list(cpg_list, region_names=region_names,
                                 make_one_indexed=make_one_indexed, is_one_indexed=False)

    @classmethod
    def from_fasta(
            cls,
            fn: str | Path,
            region_names: tuple[str] = None,
            make_one_indexed: bool = False
    ) -> Self:
        """Create a CpGIndex from a fasta file."""
        genome = Genome.from_fasta(fn)
        return cls.from_genome(genome, region_names=region_names, make_one_indexed=make_one_indexed)

    def filter_cpg_index_by_regions(
            self,
            regions: Regions,
    ) -> Self:
        """Create a new CpGIndex containing only CpG sites that fall within the given regions."""

        new_cpg_list = []
        cpg_regions:list[str] = []
        for chrm, loc in self.cpg_list:

            hit = regions.region_at_locus(chrm, loc-self.one_indexed)
            if hit:
                cpg_regions.append(hit)
                new_cpg_list.append((chrm, loc))

        return CpGIndex.from_cpg_list(
            new_cpg_list,
            region_names=tuple(cpg_regions),
            is_one_indexed=self.one_indexed,)

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
    """Class for holding and working with dictionary of sequences.

    Genome[chrm] also returns the sequence for that chromosome."""
    sequences: MappingProxyType[str, str]
    filename: str | Path = None

    def harmonise_chrm_names(self) -> Self:
        """add "chr" to chromosome names that don't have it, and remove it from them that do.

        Creates a new instance of Genome as Genome is immutable."""
        new_seqs = {k: v for k, v in self.sequences.items()}
        for k in self.sequences.keys():
            if k.startswith('chr'):
                new_seqs[k[3:]] = self.sequences[k]
            else:
                new_seqs['chr' + k] = self.sequences[k]

        return Genome.from_dict(sequences=new_seqs, filename=self.filename)

    @classmethod
    def from_fasta(cls, fn: str | Path,
                   full_desc=False,
                   cannonical_only=True) -> Self:
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


    def get_cpg_index(self, one_indexed=False) -> CpGIndex:
        """Get the CpG index for this genome."""
        return CpGIndex.from_genome(self, make_one_indexed=one_indexed)

    @property
    def chromsomes(self) -> list[str]:
        """List of chromosome names in the genome."""
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
    assert test_cindex.locus2index == MappingProxyType({('a', 3): 0, ('a', 4): 0, ('a', 5): 1, ('a', 6): 1, ('a', 9): 2, ('a', 10): 2}), (
        str({k:test_cindex.locus2index[k] for k in sorted(test_cindex.locus2index.keys())})
    )


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