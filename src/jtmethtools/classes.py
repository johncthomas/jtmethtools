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
from numpy import typing as npt
from numpy.typing import NDArray





type CpGIndexDict =  dict[str, dict[int, int]]
type Pathy = str|Path

__all__ = ['Regions', 'CpGIndex', 'Genome', 'filter_cpg_index_by_regions', 'LociRange']

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
        df = load_region_bed(filename)
        return cls.from_df(df)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """Expects columns Start, End, Name & Chrm."""
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



class CpGIndex:
    def __init__(self,
                 index:CpGIndexDict,
                 region_names:list[str]=None,
                 n_cpg: int=None,
                 ):
        """For holding map of every CpG locus to an index number.

        CpGIndex.index structured thusly:
            {chromsome:{locus:idx, ...}, ...}

        Constructor available Genome.get_cpg_index.

        Args:
            index: The mapping of chrm, loc -> cpg_idx
            region_names: regions names at each CpG. Automatically generated if
                filter_regions is given.
            n_cpg: Total number of CpGs. calculated if not given."""
        self.index = index
        self.region_names = region_names
        if n_cpg is None:
            n_cpg = sum([len(v) for v in self.index.values()])
        self.n_cpg = n_cpg

    def to_columns(
            self
    ) -> TypedDict('CpGIndexColumns',
                   {'Chrm': pd.Categorical, 'Locus': npt.NDArray}):
        """Output index in long-form as columns, "Chrm" and "Locus".
        """
        n = self.n_cpg
        codes_chrm = np.zeros(n, dtype=np.int32)
        cats_chrm = []

        loci_values = np.zeros(n, dtype=np.uint32)

        start = 0
        for i, (chrm, loci) in enumerate(self.index.items()):
            cats_chrm.append(chrm)
            end = start + len(loci)
            codes_chrm[start:end] = i
            loci_values[start:end] = list(loci.keys())
            start = end

        chromosomes = pd.Categorical.from_codes(codes_chrm, cats_chrm)

        return {'Chrm': chromosomes, 'Locus': loci_values}


    @classmethod
    def _ttest_to_columns(cls) -> None:
        cpg_index = {'1': {1: 0, 10: 1, 100: 2}, '2': {2: 0, 20: 1, 200: 2}}
        #cpg_regions = ['1_1', '1_1', '1_2', '2_1', '2_1', '2_1']

        indx = cls(cpg_index)

        cols = indx.to_columns()

        assert cols['Chrm'].tolist() == ['1', '1', '1', '2', '2', '2']

        assert np.all(
            np.array(
                [1, 10, 100, 2, 20, 200],
                dtype=cols['Locus'].dtype
            ) == cols['Locus']
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
    def from_fasta(cls, fn: str | Path, full_desc=False) -> Self:
        """Load a fasta file into a Genome object"""
        genome = cls(
            MappingProxyType(
                fasta_to_dict(fn, full_desc=full_desc)
            ),
            filename=fn
        )
        return genome

    @classmethod
    def from_dict(cls, sequences: dict[str, str], filename='<from dict>') -> Self:
        """Create a Genome object from a dict.

        Interally uses MappingProxyType for immutable dict like."""
        return cls(MappingProxyType(sequences), filename=filename)

    def __getitem__(self, key):
        return self.sequences[key]

    @cached_property
    def cpg_index(self) -> CpGIndex:
        idx = self.get_cpg_index()
        return idx

    @lru_cache(1)
    def get_cpg_index(self) -> CpGIndex:

        cpg_chrm_loc_to_idx: dict[str, dict[int, int]] = {}
        cpg_count = 0
        for chrm, contig in self.sequences.items():
            if chrm not in CANNONICAL_CHRM:
                continue
            cpg_chrm_loc_to_idx[chrm] = loc2idx = {}
            for loc in range(len(contig) - 1):
                if contig[loc] == 'C':
                    if contig[loc + 1] == 'G':
                        loc2idx[loc] = cpg_count
                        cpg_count += 1

        return CpGIndex(
            index=cpg_chrm_loc_to_idx,
            n_cpg=cpg_count
        )

    @property
    def chromsomes(self) -> list[str]:
        return list(self.sequences.keys())

    def __repr__(self):
        return f"<Genome: filename={self.filename},  id={id(self)}>"

    def __hash__(self):
        # attrs version of the hash tries to hash .sequences,
        #   which raises an error.
        return id(self)


def filter_cpg_index_by_regions(
        cpg_index:CpGIndex,
        regions:Regions
) -> CpGIndex:
    cpg_dict = {}
    n_filt_cpg = 0
    cpg_regions = []
    for chrm, fullidx in cpg_index.index.items():
        cpg_dict[chrm] = filtidx = {}
        for loc in fullidx.keys():
            hit = regions.region_at_locus(chrm, loc)
            if hit:
                filtidx[loc] = n_filt_cpg
                n_filt_cpg += 1
                cpg_regions.append(hit)

    return CpGIndex(cpg_dict, region_names=cpg_regions)



