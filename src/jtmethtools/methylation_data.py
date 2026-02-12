import collections
import copy
import json

import sys
from os import PathLike
import os
from pathlib import Path
from typing import (
    Self,
)

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
import pysam
import pyarrow as pa

from pyarrow import parquet

import pandas as pd

from jtmethtools.alignments import (
    Alignment,
    iter_bam,
    get_bismark_met_str
)

from jtmethtools.classes import *

from jtmethtools.util import (
    logger,
    CANNONICAL_CHRM
)

from jtmethtools.alignments import iter_bam_segments

logger.remove()

def table2df(table: pa.Table) -> pd.DataFrame:
    mapping = {schema.type: pd.ArrowDtype(schema.type) for schema in table.schema}

    return table.to_pandas(types_mapper=mapping.get, ignore_metadata=True)


def read_parquet(fn) -> pd.DataFrame:
    """load a parquet file and convert to a pandas dataframe.
    Works when pd.read_parquet fails on the dictionary types."""
    tls = pa.parquet.read_table(fn)
    tls = table2df(tls)
    return tls


def _load_methylation_data_reliable(datdir):
    datdir = Path(datdir)
    # logger.info(f'loading methylation data for sample {sample}')
    # datdir = bigdata/f'users/jct61/tasks/250709.pattern_count/methylation-data/ICGC_250714/{sample}'
    locdat = read_parquet(datdir / 'locus_data.parquet')
    readdat = read_parquet(datdir / 'read_data.parquet')
    with open(datdir / 'metadata.json') as f:
        metadat = json.load(f)

    return locdat, readdat, metadat


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


class MethylationDataset:
    def __init__(
            self,
            locus_data:pd.DataFrame,
            read_data:pd.DataFrame,
            processes:list[dict]=None,
            name:str=None,
    ):
        self.locus_data = locus_data
        self.read_data = read_data
        self.processes = [p for p in (processes or []) if p is not None]
        self.name = name

    @property
    def metadata(self) -> list[dict]:
        logger.warning("DEPRECIATION: metadata is now a list of processes. Use the .processes property to access it.")
        return self.processes

    @classmethod
    def from_dir(cls, datdir:Path|str) -> Self:
        datdir = Path(datdir)
        locdat, readdat, processes = load_methylation_data(datdir)
        return cls(
            locus_data=locdat,
            read_data=readdat,
            processes=processes,
            name=datdir.name,
        )

    def write_to_dir(self, outdir:Path|str, create_outdir=True) -> None:
        outdir = Path(outdir)
        if create_outdir:
            outdir.mkdir(parents=True, exist_ok=True)
        self.locus_data.to_parquet(outdir/'locus_data.parquet')
        self.read_data.to_parquet(outdir/'read_data.parquet')
        with open(outdir/'metadata.json', 'w') as f:
            json.dump(self.processes, f, indent=4)

    def drop_methylated_CH(self) -> Self:
        """Return a new MethylationDataset with reads that have methylated CpH
        removed from locus_data and read_data."""

        m = self.locus_data.BismarkCode.isin(['X', 'H', 'U'])
        bad_aln = self.locus_data.loc[m, 'AlignmentIndex'].unique()
        locdat = self.locus_data.loc[~self.locus_data.AlignmentIndex.isin(bad_aln)].reset_index(drop=True)
        readat = self.read_data.loc[~self.read_data.AlignmentIndex.isin(bad_aln)].reset_index(drop=True)
        procs = copy.copy(self.processes)
        procs.append({
            'name': 'drop_methylated_CH',
            'date_time': str(pd.Timestamp.now()),
            'n_dropped_reads': len(bad_aln),
            'reads_remaining': readat.shape[0],
        })

        return MethylationDataset(
            locus_data=locdat,
            read_data=readat,
            processes=procs,
        )

    # getitem to allow tuple unpacking, e.g. locus, read, meta = dataset
    def __getitem__(self, index):
        if index == 0:
            return self.locus_data
        elif index == 1:
            return self.read_data
        elif index == 2:
            return self.processes
        else:
            raise IndexError("MethylationDataset only has three items: locus_data, read_data, metadata.")



def load_methylation_data(datdir) -> MethylationDataset:
    datdir = Path(datdir)
    try:
        locdat = pd.read_parquet(datdir / 'locus_data.parquet')
        readdat = pd.read_parquet(datdir / 'read_data.parquet')
    except ValueError:
        # not sure why we sometimes get the ValueError: Dictionary type not supported error
        locdat, readdat, metadat = _load_methylation_data_reliable(datdir)
    with open(datdir / 'metadata.json') as f:
        metadat = json.load(f)

    return MethylationDataset(locdat, readdat, metadat)


def process_bam_methylation_data(
        bamfn:Path|str,
        first_i:int=0,
        last_i:int=np.inf,
        regions:Regions=None,
        paired_end=True,
        cannonical_chrm_only=True,
        include_unmethylated_ch=False,
        chunk_size=int(1e6),
        min_mapq=0,
        #min_phred=0, # this one's easier at the table level
        drop_methylated_ch_reads=False,
) -> MethylationDataset:

    logger.info(f"Processing BAM, {bamfn}, {first_i}-{last_i}.")

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

    # 1. Chromosome mapping
    with pysam.AlignmentFile(bamfn) as af:
        header = af.header.to_dict()
        chrm_map = {}
        chrm_i = 0
        for s in header['SQ']:
            ref = s['SN']
            if (not cannonical_chrm_only) or (ref in CANNONICAL_CHRM):
                chrm_map[ref] = chrm_i
                chrm_i += 1
    if len(chrm_map) > 255:
        chrm_dtype = np.int16
    else:
        chrm_dtype = np.int8


    # 2. nucleotide mapping
    nt_map = {n: i for i, n in enumerate('NACGT')}

    # 3. Bismark code mapping
    bismark_map = {n: i for i, n in enumerate('zZhHxXuU.')}

    dictionary_mappings = {'ReadNucleotide': nt_map,
                            'BismarkCode': bismark_map,
                            'Chrm': chrm_map}

    if regions is None:
        filtering_by_region = False
    else:
        filtering_by_region = True

    if cannonical_chrm_only and (not filtering_by_region):
        def include_alignment(a) -> bool:
            return a.reference_name in CANNONICAL_CHRM
    elif cannonical_chrm_only and filtering_by_region:
        def include_alignment(a) -> bool:
            return (a.reference_name in CANNONICAL_CHRM) and (len(a.get_hit_regions(regions)) > 0)
    elif not cannonical_chrm_only and filtering_by_region:
        def include_alignment(a) -> bool:
            return len(a.get_hit_regions(regions)) > 0
    elif not cannonical_chrm_only and not filtering_by_region:
        def include_alignment(a) -> bool:
            return True
    else:
        raise ValueError("Logic error in include_alignment definition. This should not happen.")


    if last_i == np.inf:
        # count em if we aren't processing a subset of the bam
        i = 0
        for i, aln in enumerate(iter_bam_segments(bamfn, paired_end)):
            pass
        logger.info(f'Reads in BAM: {i}')
        max_reads = i+1
    else:
        max_reads = last_i - first_i

    # note: types that will be dictionary encoded in the Table need to use unsigned integers
    #  so that they can be converted to pandas DataFrame later
    read_data = {
        'AlignmentIndex': np.zeros(max_reads, dtype=np.uint32),
        'Start': np.zeros(max_reads, dtype=np.uint32),
        'End': np.zeros(max_reads, dtype=np.uint32),
        'MappingQuality': np.zeros(max_reads, dtype=np.uint8),
        'Chrm': np.zeros(max_reads, dtype=chrm_dtype), # dictionary
    }

    def get_empty_locus_chunk() -> dict[str, NDArray]:
        return {
            'AlignmentIndex': np.zeros(chunk_size, dtype=np.uint32),
            'ReadNucleotide': np.zeros(chunk_size, dtype=np.int8), # dictionary
            'PhredScore': np.zeros(chunk_size, dtype=np.uint8),
            'Chrm': np.zeros(chunk_size, dtype=chrm_dtype),  # dictionary
            'Position': np.zeros(chunk_size, dtype=np.uint32),
            'ReadPostion': np.zeros(chunk_size, dtype=np.uint32),
            'BismarkCode': np.zeros(chunk_size, dtype=np.int8),  # dictionary
        }

    def get_locus_table_from_chunk(loc_data:dict, locus_cursor:int) -> pa.Table:
        # save the current locus data chunk
        # convert arrays to pa.Table
        writable_data = {}
        for k, array in loc_data.items():
            array = array[:locus_cursor]
            # convert the dictionary types
            if k in dictionary_mappings.keys():
                writable_data[k] = mapping_to_pa_dict(dictionary_mappings[k], array)
            else:
                writable_data[k] = pa.array(array)
        loc_table = pa.Table.from_pydict(writable_data, )
        return loc_table

    locus_data_tables = []
    current_locus_data = get_empty_locus_chunk()
    locus_cursor = 0
    read_cursor = 0
    logger.info('At start for processing alignments {')
    log_memory_footprint()
    for aln_i, aln in enumerate(iter_bam(bamfn, paired_end=paired_end)):
        if aln_i < first_i:
            continue
        if aln_i >= last_i:
            break

        aln_i += 1 # AlignmentIndex 0 is for empty rows.

        aln:Alignment

        if not include_alignment(aln):
            continue

        if aln.mapping_quality() < min_mapq:
            continue
        if aln.a.is_unmapped:
            continue
        metstr = aln.metstr
        if not metstr:
            # malformed bismark data can result in empty metstr
            #   for otherwise valid alignments
            continue
        if drop_methylated_ch_reads and ('H' in metstr or 'X' in metstr or 'U' in metstr):
            continue

        data_added = False
        for pos, met in aln.locus_methylation.items():
            if met == '.':
                continue
            if include_unmethylated_ch or met.isupper() or (met.lower() == 'z'):
                data_added = True
                # record the locus data
                current_locus_data['AlignmentIndex'][locus_cursor] = aln_i
                current_locus_data['ReadNucleotide'][locus_cursor] = nt_map[aln.locus_nucleotide[pos]]
                current_locus_data['PhredScore'][locus_cursor] = aln.locus_quality[pos]
                current_locus_data['Chrm'][locus_cursor] = chrm_map[aln.reference_name]
                current_locus_data['Position'][locus_cursor] = pos
                current_locus_data['BismarkCode'][locus_cursor] = bismark_map[met]
                current_locus_data['ReadPostion'][locus_cursor] = pos-aln.reference_start
                #current_locus_data['IsForward'][locus_cursor] = aln.is_forward

                locus_cursor += 1

                if locus_cursor >= chunk_size:
                    loc_table = get_locus_table_from_chunk(current_locus_data, locus_cursor)
                    locus_data_tables.append(loc_table)
                    current_locus_data = get_empty_locus_chunk()
                    locus_cursor = 0
        if data_added:
            # record the read data
            read_data['AlignmentIndex'][read_cursor] = aln_i
            read_data['Start'][read_cursor] = aln.reference_start
            read_data['End'][read_cursor] = aln.reference_end
            read_data['MappingQuality'][read_cursor] = aln.mapping_quality()
            read_data['Chrm'][read_cursor] = chrm_map[aln.reference_name]
            read_cursor += 1

            if read_cursor >= max_reads:
                logger.info(f'Reached max reads, stopping processing. {bamfn=}, {max_reads=}.')

    # save the last locus data chunk
    loc_table = get_locus_table_from_chunk(current_locus_data, locus_cursor)
    locus_data_tables.append(loc_table)
    locus_table = pa.concat_tables(locus_data_tables)

    # create the read table
    read_data_final = {}
    for k, array in read_data.items():
        array = array[:read_cursor]
        if k in dictionary_mappings.keys():
            # convert the dictionary types
            read_data_final[k] = mapping_to_pa_dict(dictionary_mappings['Chrm'], array)
        else:
            read_data_final[k] = pa.array(array)

    read_table = pa.Table.from_pydict(read_data_final, )

    logger.info('Finished processing alignments.')
    log_memory_footprint()

    process = {
        'name': 'process_bam_methylation_data',
        'bam_file': str(bamfn),
         'regions_file': str(regions) if regions is not None else None,
         'date_time': str(pd.Timestamp.now()),
    }
    locus_data = table2df(locus_table)
    read_data = table2df(read_table)
    processes = [process]

    locus_data = locus_data.loc[locus_data.AlignmentIndex != 0].reset_index(drop=True)
    read_data = read_data.loc[read_data.AlignmentIndex != 0].reset_index(drop=True)

    locus_data.loc[:, 'MetCpG'] = locus_data.BismarkCode == 'Z'

    return MethylationDataset(
        locus_data=locus_data, read_data=read_data,
        processes=processes
    )


def write_methylation_tables(
        outdir:Path|str,
        locus_table:pa.Table,
        read_table:pa.Table,
        metadata:dict=None,
        create_outdir:bool=True,
) -> None:
    """Write the methylation dataset to a directory."""
    outdir = Path(outdir)
    if not outdir.exists() and create_outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    # write the locus data
    locus_fn = outdir / 'locus_data.parquet'
    parquet.write_table(locus_table, str(locus_fn))

    # write the read data
    read_fn = outdir / 'read_data.parquet'
    parquet.write_table(read_table, str(read_fn))

    # write metadata
    metadata_fn = outdir / 'metadata.json'
    with open(metadata_fn, 'w') as f:
        json.dump(metadata, f, indent=4)



def sample_reads(
        data:MethylationDataset,
        n_reads:int,
        with_replacement=False,
        seed:int=None,
) -> MethylationDataset:
    """Sample reads from a methylation dataset."""
    from copy import copy
    locus_table, read_table, metadata = data.locus_data.copy(), data.read_data.copy(), copy(data.metadata)

    for t in locus_table, read_table:
        t.drop('__index_level_0__', axis=1, inplace=True, errors='ignore')

    read_table.set_index('AlignmentIndex', inplace=True, drop=False)
    locus_table = locus_table.loc[locus_table.AlignmentIndex.isin(read_table.AlignmentIndex)]

    if seed is not None:
        np.random.seed(seed)

    sampled_alnidx = np.random.choice(
        read_table.AlignmentIndex, size=n_reads, replace=with_replacement,
    )

    read_groups = locus_table.groupby('AlignmentIndex').groups

    resamp_read_table = read_table.loc[sampled_alnidx].reset_index(drop=True)
    resamp_read_table.loc[:, 'OldAlnIdx'] = resamp_read_table.AlignmentIndex
    resamp_read_table.loc[:, 'AlignmentIndex'] = resamp_read_table.index

    loc_rows = []
    new_loc_alnIdx = []

    for newidx, row in resamp_read_table.iterrows():
        loc_rows_idx = read_groups[row.OldAlnIdx]
        loc_rows.extend(loc_rows_idx)
        new_loc_alnIdx.extend([newidx] * len(loc_rows_idx))

    resamp_locus_table = locus_table.loc[loc_rows]
    resamp_locus_table.loc[:, 'AlignmentIndex'] = new_loc_alnIdx

    resamp_read_table.drop('OldAlnIdx', axis=1, inplace=True)

    metadata.get('processes', []).append(
        f'sample_reads: sampled {n_reads} with_replacement={with_replacement}'
    )

    methylation_data = MethylationDataset(
        locus_data=resamp_locus_table,
        read_data=resamp_read_table,
        processes=metadata
    )

    return methylation_data



def synthetic_sample(
        inputs:Iterable[tuple[str|Path, float, bool]],
        target_reads:int,
) -> MethylationDataset:
    """Create a synthetic methylation dataset by combining multiple datasets.

    Inputs is a list of tuples of (data_dir, proportion, with_replacement).

    Total reads might end up slightly different due to rounding."""

    logger.info(f"Creating synthetic sample with target reads: {target_reads}")

    locus_tables = []
    read_tables = []
    total_reads = 0
    metadata = {
        'processes': ['create_synthetic_sample: ' + ', '.join([f'{p} from {d}' for d, p, _ in inputs])]
    }

    # normalise the proportions
    total_proportion = sum([p for _, p, _ in inputs])
    inputs = [(d, p / total_proportion, w) for d, p, w in inputs]

    # log proportions
    logger.info("Creating synthetic sample with the following inputs:\n" +
                '\n'.join([f"  - {p:.1%} from {d} (with_replacement={w})" for d, p, w in inputs]))

    for datdir, proportion, with_replacement in inputs:
        data = MethylationDataset.from_dir(datdir)
        n_reads = int(round(target_reads * proportion, 0))
        sampled_data = sample_reads(data, n_reads, with_replacement=with_replacement)

        # adjust AlignmentIndex in locus data
        read_table = sampled_data.read_data
        locus_table = sampled_data.locus_data

        read_table.set_index('AlignmentIndex', inplace=True, drop=False)
        locus_table = locus_table.loc[locus_table.AlignmentIndex.isin(read_table.AlignmentIndex)]

        alnidx_map = {
            old_idx: new_idx for new_idx, old_idx
            in enumerate(read_table.AlignmentIndex, start=total_reads)
        }

        locus_table.loc[:, 'AlignmentIndex'] = locus_table.AlignmentIndex.map(alnidx_map)
        read_table.loc[:, 'AlignmentIndex'] = read_table.AlignmentIndex.map(alnidx_map)

        locus_tables.append(locus_table)
        read_tables.append(read_table)

        total_reads += len(read_table)

    combined_locus_table = pd.concat(locus_tables, ignore_index=True)
    combined_read_table = pd.concat(read_tables, ignore_index=True)

    synthetic_data = MethylationDataset(
        locus_data=combined_locus_table,
        read_data=combined_read_table,
        processes=metadata
    )

    logger.info(f'Finished creating synthetic sample. Actual total reads: {total_reads}')

    return synthetic_data


def _generate_test_dataset() -> tuple[Path, Path, MethylationDataset, MethylationDataset]:
    rdat = {'AlignmentIndex': {8: 8, 12: 12, 13: 13, 1866: 1866, 1867: 1867},
            'Start': {8: 15995, 12: 16015, 13: 16019, 1866: 605302, 1867: 605305},
            'End': {8: 16145, 12: 16164, 13: 16168, 1866: 605450, 1867: 605455},
            'MappingQuality': {8: 31, 12: 31, 13: 31, 1866: 39, 1867: 39},
            'Chrm': {8: '1', 12: '1', 13: '1', 1866: '1', 1867: '1'},
            'IsForward': {8: False, 12: False, 13: False, 1866: False, 1867: False}}
    test_read_table = pd.DataFrame.from_dict(rdat).reset_index(
        drop=True)  # shouldn't be dependent on the index == AlignmentIndex

    ldat = {
        'AlignmentIndex': {0: 8, 1: 8, 2: 8, 3: 12, 4: 12, 5: 12, 6: 12, 7: 12, 8: 13, 9: 13, 10: 13, 11: 13, 12: 13,
                           4339: 1866, 4340: 1866, 4341: 1866, 4342: 1866, 4343: 1867, 4344: 1867, 4345: 1867,
                           4346: 1867},
        'ReadNucleotide': {0: 'G', 1: 'G', 2: 'G', 3: 'G', 4: 'G', 5: 'G', 6: 'G', 7: 'G', 8: 'G', 9: 'G', 10: 'G',
                           11: 'G', 12: 'G', 4339: 'A', 4340: 'A', 4341: 'A', 4342: 'A', 4343: 'A', 4344: 'A',
                           4345: 'A', 4346: 'A'},
        'PhredScore': {0: 41, 1: 41, 2: 41, 3: 41, 4: 41, 5: 37, 6: 41, 7: 41, 8: 41, 9: 41, 10: 41, 11: 41, 12: 41,
                       4339: 41, 4340: 37, 4341: 32, 4342: 37, 4343: 41, 4344: 41, 4345: 37, 4346: 41},
        'Chrm': {0: '1', 1: '1', 2: '1', 3: '1', 4: '1', 5: '1', 6: '1', 7: '1', 8: '1', 9: '1', 10: '1', 11: '1',
                 12: '1', 4339: '1', 4340: '1', 4341: '1', 4342: '1', 4343: '1', 4344: '1', 4345: '1', 4346: '1'},
        'Position': {0: 16057, 1: 16069, 2: 16081, 3: 16057, 4: 16069, 5: 16081, 6: 16138, 7: 16140, 8: 16057, 9: 16069,
                     10: 16081, 11: 16138, 12: 16140, 4339: 605329, 4340: 605369, 4341: 605419, 4342: 605427,
                     4343: 605329, 4344: 605369, 4345: 605419, 4346: 605427},
        'BismarkCode': {0: 'Z', 1: 'Z', 2: 'Z', 3: 'Z', 4: 'Z', 5: 'Z', 6: 'Z', 7: 'Z', 8: 'Z', 9: 'Z', 10: 'Z',
                        11: 'Z', 12: 'Z', 4339: 'z', 4340: 'z', 4341: 'z', 4342: 'z', 4343: 'z', 4344: 'z', 4345: 'z',
                        4346: 'z'},
        'PosFromLeft': {0: 63, 1: 75, 2: 87, 3: 43, 4: 55, 5: 67, 6: 124, 7: 126, 8: 39, 9: 51, 10: 63, 11: 120,
                        12: 122, 4339: 28, 4340: 68, 4341: 118, 4342: 126, 4343: 25, 4344: 65, 4345: 115, 4346: 123},
        'PosFromRight': {0: 87, 1: 75, 2: 63, 3: 106, 4: 94, 5: 82, 6: 25, 7: 23, 8: 110, 9: 98, 10: 86, 11: 29, 12: 27,
                         4339: 120, 4340: 80, 4341: 30, 4342: 22, 4343: 125, 4344: 85, 4345: 35, 4346: 27},
        'CH': {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False,
               10: False, 11: False, 12: False, 4339: False, 4340: False, 4341: False, 4342: False, 4343: False,
               4344: False, 4345: False, 4346: False},
        'MetCpG': {0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True, 10: True,
                   11: True, 12: True, 4339: False, 4340: False, 4341: False, 4342: False, 4343: False, 4344: False,
                   4345: False, 4346: False},
        'IsCpG': {0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True, 10: True,
                  11: True, 12: True, 4339: True, 4340: True, 4341: True, 4342: True, 4343: True, 4344: True,
                  4345: True, 4346: True}}
    test_locus_table = pd.DataFrame.from_dict(ldat).reset_index(drop=True)

    metadata = {'processes': ["Random small dataset for testing sample_reads"]}

    test_data1 = MethylationDataset(
        locus_data=test_locus_table,
        read_data=test_read_table,
        processes=metadata
    )

    rdat2, ldat2 = (
        {'AlignmentIndex': {1698: 1698, 1695: 1695, 898: 898, 528: 528, 943: 943},
         'Start': {1698: 597730, 1695: 597726, 898: 138669, 528: 134898, 943: 138748},
         'End': {1698: 597880, 1695: 597854, 898: 138819, 528: 135040, 943: 138898},
         'MappingQuality': {1698: 38, 1695: 37, 898: 31, 528: 31, 943: 31},
         'Chrm': {1698: '1', 1695: '1', 898: '1', 528: '1', 943: '1'},
         'IsForward': {1698: True, 1695: True, 898: False, 528: True, 943: True}},
        {'AlignmentIndex': {847: 528, 848: 528, 849: 528, 2362: 898, 2363: 898, 2417: 943, 3708: 1695, 3709: 1695,
                            3710: 1695, 3711: 1695, 3712: 1695, 3713: 1695, 3714: 1695, 3728: 1698, 3729: 1698,
                            3730: 1698, 3731: 1698, 3732: 1698, 3733: 1698, 3734: 1698, 3735: 1698},
         'ReadNucleotide': {847: 'C', 848: 'T', 849: 'C', 2362: 'G', 2363: 'G', 2417: 'C', 3708: 'C', 3709: 'C',
                            3710: 'C', 3711: 'C', 3712: 'C', 3713: 'C', 3714: 'T', 3728: 'C', 3729: 'C', 3730: 'C',
                            3731: 'C', 3732: 'C', 3733: 'C', 3734: 'C', 3735: 'C'},
         'PhredScore': {847: 41, 848: 41, 849: 41, 2362: 41, 2363: 41, 2417: 41, 3708: 41, 3709: 41, 3710: 41, 3711: 41,
                        3712: 41, 3713: 41, 3714: 41, 3728: 41, 3729: 41, 3730: 37, 3731: 41, 3732: 41, 3733: 41,
                        3734: 41, 3735: 41},
         'Chrm': {847: '1', 848: '1', 849: '1', 2362: '1', 2363: '1', 2417: '1', 3708: '1', 3709: '1', 3710: '1',
                  3711: '1', 3712: '1', 3713: '1', 3714: '1', 3728: '1', 3729: '1', 3730: '1', 3731: '1', 3732: '1',
                  3733: '1', 3734: '1', 3735: '1'},
         'Position': {847: 134998, 848: 135027, 849: 135030, 2362: 138720, 2363: 138780, 2417: 138780, 3708: 597746,
                      3709: 597781, 3710: 597793, 3711: 597806, 3712: 597817, 3713: 597839, 3714: 597853, 3728: 597781,
                      3729: 597793, 3730: 597806, 3731: 597817, 3732: 597839, 3733: 597853, 3734: 597862, 3735: 597864},
         'BismarkCode': {847: 'Z', 848: 'z', 849: 'Z', 2362: 'Z', 2363: 'Z', 2417: 'Z', 3708: 'Z', 3709: 'Z', 3710: 'Z',
                         3711: 'Z', 3712: 'Z', 3713: 'Z', 3714: 'z', 3728: 'Z', 3729: 'Z', 3730: 'Z', 3731: 'Z',
                         3732: 'Z', 3733: 'Z', 3734: 'Z', 3735: 'Z'},
         'PosFromLeft': {847: 100, 848: 129, 849: 132, 2362: 52, 2363: 112, 2417: 32, 3708: 20, 3709: 55, 3710: 67,
                         3711: 80, 3712: 91, 3713: 113, 3714: 127, 3728: 51, 3729: 63, 3730: 76, 3731: 87, 3732: 109,
                         3733: 123, 3734: 132, 3735: 134},
         'PosFromRight': {847: 42, 848: 13, 849: 10, 2362: 98, 2363: 38, 2417: 118, 3708: 108, 3709: 73, 3710: 61,
                          3711: 48, 3712: 37, 3713: 15, 3714: 1, 3728: 99, 3729: 87, 3730: 74, 3731: 63, 3732: 41,
                          3733: 27, 3734: 18, 3735: 16},
         'CH': {847: False, 848: False, 849: False, 2362: False, 2363: False, 2417: False, 3708: False, 3709: False,
                3710: False, 3711: False, 3712: False, 3713: False, 3714: False, 3728: False, 3729: False, 3730: False,
                3731: False, 3732: False, 3733: False, 3734: False, 3735: False},
         'MetCpG': {847: True, 848: False, 849: True, 2362: True, 2363: True, 2417: True, 3708: True, 3709: True,
                    3710: True, 3711: True, 3712: True, 3713: True, 3714: False, 3728: True, 3729: True, 3730: True,
                    3731: True, 3732: True, 3733: True, 3734: True, 3735: True},
         'IsCpG': {847: True, 848: True, 849: True, 2362: True, 2363: True, 2417: True, 3708: True, 3709: True,
                   3710: True, 3711: True, 3712: True, 3713: True, 3714: True, 3728: True, 3729: True, 3730: True,
                   3731: True, 3732: True, 3733: True, 3734: True, 3735: True}}
    )

    test_data2 = MethylationDataset(
        locus_data=pd.DataFrame.from_dict(ldat2),
        read_data=pd.DataFrame.from_dict(rdat2),
        processes=metadata,
    )

    od1 = Path.home() / 'tmp/test_mdat1_hroqwei'
    od2 = Path.home() / 'tmp/test_mdat2_fhboqwei'
    od1.mkdir(exist_ok=True)
    od2.mkdir(exist_ok=True)

    test_data1.write_to_dir(od1)
    test_data2.write_to_dir(od2)

    return od1, od2, test_data1, test_data2

def ttest_sample_reads():
    logger.add(sys.stdout, level='INFO')

    # create small test datasets, func returns paths.
    od1, od2, test_data1, test_data2 = _generate_test_dataset()

    d = synthetic_sample(
        [
            (od1, 0.8, True),
            (od2, 0.2, True)
        ],
        5000
    )

    import shutil
    shutil.rmtree(od1)
    shutil.rmtree(od2)

    ratio = d.read_data.Start.isin(test_data1.read_data.Start.values).sum() / d.read_data.shape[0]
    assert abs(ratio - 0.8) < 0.001
    assert abs(d.read_data.shape[0] - 5000) < 5

    logger.info("ttest_sample_reads passed.")


def write_coverage(locus_data:pd.DataFrame, outfile:str|Path) -> None:
    """Write a coverage file from locus data."""
    outfile = Path(outfile)
    if not locus_data.IsCpG.all():
        locus_data = locus_data.loc[locus_data.IsCpG]

    if "Chrm" not in locus_data.columns:
        locus_data = locus_data.reset_index()

    ld = locus_data.head(10000)

    g = ld.groupby(['Chrm', 'Position'])
    m = g.MetCpG.sum()
    n = g.MetCpG.count()

    cov = m.reset_index()
    cov.loc[:, 'Unmet'] = (n - m).values
    cov.insert(2, 'Perc', (m / n).apply(lambda x: int(round(x * 100))).values)
    cov.insert(1, 'Pos2', cov.Position)

    # create dir
    outfile.parent.mkdir(parents=True, exist_ok=True)

    cov.to_csv(
        outfile,
        sep='\t', header=None, index=False,
    )


def cli_synthetic_sample(clargs=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Create a synthetic methylation dataset by combining multiple datasets. "
                    "Example "
    )

    parser.add_argument(
        "--inputs", "-i",
        nargs=3,
        action="append",
        metavar=("PATH", "PROPORTION", "WITH_REPLACEMENT"),
        required=True,
        help=(
            "Input dataset directory, proportion (float), whether to sample with "
            "replacement ('R'/'N')."
        ),
    )

    parser.add_argument(
        "--total-reads", "-n",
        type=int,
        required=True,
        help="Target number of reads in the synthetic dataset. May differ slightly due to rounding.",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for the synthetic dataset.",
    )

    parser.add_argument(
        "--coverage-out", "-c",
        type=Path,
        default=None,
        help=(
            "Optional output file for coverage data (Bismark BED format). "
            "Using *.cov.gz extension is recommended."
        ),
    )

    if clargs is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(clargs)

    inputs = []
    for datdir, prop, replacement in args.inputs:
        proportion = float(prop)
        replacement = replacement.upper()
        if not replacement.startswith(('R', 'N')):
            raise ValueError("WITH_REPLACEMENT must be 'R' for Replacement or 'N' for Not.")

        with_replacement = True if replacement.startswith('R') else False

        inputs.append((datdir, proportion, with_replacement))


    synthetic_data = synthetic_sample(
        inputs=inputs,
        target_reads=args.total_reads
    )

    if args.coverage_out is not None:
        logger.info(f"Writing coverage data to {args.coverage_out}")
        write_coverage(synthetic_data.locus_data, args.coverage_out)

    logger.info(f"Writing synthetic dataset to {args.output_dir}")

    synthetic_data.write_to_dir(args.output_dir)

def ttest_cli_synthetic_sample():
    logger.add(sys.stdout, level='INFO')

    od1, od2, test_data1, test_data2 = _generate_test_dataset()

    outdir = Path.home()/'tmp/synthetic_mdat_fhboqwei2'

    # run the CLI function
    cli_synthetic_sample([
        '-i', str(od1), '0.7', 'R',
        '-i', str(od2), '0.3', 'R',
        '-n', '5000',
        '-o', str(outdir),
        '-c', str(outdir/'coverage.cov.gz'),
    ])

    # check the output
    data = MethylationDataset.from_dir(outdir)
    ratio = data.read_data.Start.isin(
        MethylationDataset.from_dir(od1).read_data.Start.values).sum() / data.read_data.shape[0]
    assert abs(ratio - 0.7) < 0.001
    assert abs(data.read_data.shape[0] - 5000) < 5


    logger.info("test_cli_synthetic_sample passed.")


def ttest_process_bam():
    # alignment index, from the order of this table, is used in tests, so be careful if removing
    #   rows etc.
    # Remember: AlignmentIndex is 1-based.
    samstr = """@HD	VN:1.0	SO:none
@SQ	SN:1	LN:248956422
mapq20Unorder	18	1	100	20	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
unmethRev	18	1	100	42	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
methRev	18	1	101	32	10M	*	0	10	CCCCCCCCCC	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:HHHHHHHHHH
unmethFor	2	1	102	27	10M	*	0	10	TTTTTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:zzzzzzzzzz
methFor	2	1	103	22	10M	*	0	10	CCCCCCCCCC	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:ZZZZZZZZZZ
allA	2	1	104	17	10M	*	0	10	AAAAAAAAAA	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allC	2	1	105	12	10M	*	0	10	CCCCCCCCCC	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allG	2	1	106	12	10M	*	0	10	GGGGGGGGGG	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allT	2	1	107	12	10M	*	0	10	TTTTTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
oneCpG	2	1	107	12	10M	*	0	10	TTCGTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..z.......
twoCpG	2	1	107	12	10M	*	0	10	TTCGCGTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..z.z.....
mapq19	18	1	100	19	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
mapq20	18	1	100	20	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
oneCHH\t2\t1\t104\t22\t10M\t*\t0\t10\tTTCTTTTTTT\tABCDEFGHIJ\tNM:i:tag0\tMD:Z:tag1\tXM:Z:..H.......
twoCHH\t2\t1\t104\t22\t10M\t*\t0\t10\tTTCTTCTTTT\tABCDEFGHIJ\tNM:i:tag0\tMD:Z:tag1\tXM:Z:..H..H....
twoCHH1z\t2\t1\t104\t22\t10M\t*\t0\t10\tTTCTTCTTTC\tABCDEFGHIJ\tNM:i:tag0\tMD:Z:tag1\tXM:Z:..H..H...z
"""

    samfn = Path.home() / 'tmp/test_mdat_sam_vcoqhiwb.sam'

    with open(samfn, 'w') as f:
        f.write(samstr)

    data1 = process_bam_methylation_data(
        bamfn=samfn,
        include_unmethylated_ch=True,
        paired_end=False,
    )

    # print('include_unmethylated_ch=True')
    # print(data1.read_data)
    # print(data1.locus_data.AlignmentIndex.value_counts().sort_index())

    # no reads that have a C should be removed.
    n_aln = 0
    with open(samfn, 'r') as f:
        for aln in pysam.AlignmentFile(f):
            if set(get_bismark_met_str(aln)) != {'.'}:
                n_aln += 1
    ld1 = data1.locus_data
    assert data1.read_data.shape[0] == n_aln
    assert set(data1.read_data.AlignmentIndex) == set(data1.locus_data.AlignmentIndex), (set(data1.read_data.AlignmentIndex), set(data1.locus_data.AlignmentIndex))
    assert (ld1.loc[ld1.AlignmentIndex == 13, 'BismarkCode'] == 'h').all()
    assert ld1.loc[ld1.AlignmentIndex == 16, 'BismarkCode'].value_counts()['H'] == 2
    assert ld1.loc[ld1.AlignmentIndex == 16, 'BismarkCode'].value_counts()['z'] == 1

    # options same as data1, except super short chunk size to test chunking logic.
    data1b = process_bam_methylation_data(
        bamfn=samfn,
        include_unmethylated_ch=True,
        paired_end=False,
        chunk_size=3,
    )
    assert (data1b.locus_data == data1.locus_data).all().all()

    data2 = process_bam_methylation_data(
        bamfn=samfn,
        include_unmethylated_ch=False,
        drop_methylated_ch_reads=True,
        paired_end=False,
    )

    rd2 = data2.read_data
    ld2 = data2.locus_data

    assert not (rd2.AlignmentIndex == 13).any()
    assert not (rd2.AlignmentIndex == 16).any()

    # print('include_unmethylated_ch=False, drop_methylated_ch_reads=True')
    # print(data2.read_data)
    # print(data2.locus_data.AlignmentIndex.value_counts().sort_index())


    data3 = process_bam_methylation_data(
        bamfn=samfn,
        include_unmethylated_ch=True,
        drop_methylated_ch_reads=True,
        min_mapq=20,
        paired_end=False,
    )

    # print('min_mapq=20, include_unmethylated_ch=True, drop_methylated_ch_reads=True')
    # print(data3.read_data)
    #print(data2.locus_data.AlignmentIndex.value_counts().sort_index())

    rd3 = data3.read_data
    ld3 = data3.locus_data
    # just checking mapq19 is dropped, mapq20 is kept.
    assert not (rd3.AlignmentIndex == 12).any()
    assert (rd3.AlignmentIndex == 13).any()

    print('bam processing testing complete')



#ttest_process_bam()