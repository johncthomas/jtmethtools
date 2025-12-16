import collections
import json
import pathlib
from os import PathLike
import os
from pathlib import Path
from typing import (
    Self,
    Iterator,
    Iterable, Tuple
)

import numpy as np
from numpy.typing import NDArray
import pysam
import pyarrow as pa

from pyarrow import parquet

import pandas as pd

from jtmethtools.alignments import (
    Alignment,
    iter_bam,
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
            metadata:dict,
            name:str=None,
    ):
        self.locus_data = locus_data
        self.read_data = read_data
        self.metadata = metadata
        self.name = name

    @classmethod
    def from_dir(cls, datdir:Path|str) -> Self:
        datdir = Path(datdir)
        locdat, readdat, metadat = load_methylation_data(datdir)
        return cls(
            locus_data=locdat,
            read_data=readdat,
            metadata=metadat,
            name=datdir.name,
        )

    def write_to_dir(self, outdir:Path|str, create_outdir=True) -> None:
        outdir = Path(outdir)
        if create_outdir:
            outdir.mkdir(parents=True, exist_ok=True)
        self.locus_data.to_parquet(outdir/'locus_data.parquet')
        self.read_data.to_parquet(outdir/'read_data.parquet')
        with open(outdir/'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)

    # getitem to allow tuple unpacking, e.g. locus, read, meta = dataset
    def __getitem__(self, index):
        if index == 0:
            return self.locus_data
        elif index == 1:
            return self.read_data
        elif index == 2:
            return self.metadata
        else:
            raise IndexError("MethylationDataset only has three items: locus_data, read_data, metadata.")



def load_methylation_data(datdir):
    datdir = Path(datdir)
    try:
        locdat = pd.read_parquet(datdir / 'locus_data.parquet')
        readdat = pd.read_parquet(datdir / 'read_data.parquet')
    except ValueError:
        # not sure why we sometimes get the ValueError: Dictionary type not supported error
        locdat, readdat, metadat = _load_methylation_data_reliable(datdir)
    with open(datdir / 'metadata.json') as f:
        metadat = json.load(f)

    return locdat, readdat, metadat


def process_bam_methylation_data(
        bamfn:Path|str,
        first_i:int=0,
        last_i:int=np.inf,
        regions:Regions=None,
        paired_end=True,
        cannonical_chrm_only=True,
        include_unmethylated_ch=False,
        chunk_size=int(1e6),
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

        aln:Alignment

        if not include_alignment(aln):
            continue

        metstr = aln.metstr
        if not metstr:
            # malformed bismark data can result in empty metstr
            #   for otherwise valid alignments
            continue

        # record the read data
        read_data['AlignmentIndex'][read_cursor] = aln_i
        read_data['Start'][read_cursor] = aln.reference_start
        read_data['End'][read_cursor] = aln.reference_end
        read_data['MappingQuality'][read_cursor] = aln.mapping_quality()
        read_data['Chrm'][read_cursor] = chrm_map[aln.reference_name]
        read_cursor += 1

        len_vals = len(metstr) if include_unmethylated_ch else len(metstr.replace('.', ''))

        if (locus_cursor+len_vals) >= chunk_size:
            if len_vals > chunk_size:
                raise ValueError("Chunk size exceeded by a single alignment.")

            loc_table = get_locus_table_from_chunk(current_locus_data, locus_cursor)
            locus_data_tables.append(loc_table)

            # reset the current locus data
            current_locus_data = get_empty_locus_chunk()
            locus_cursor = 0

            # # memory check
            # logger.info(f'After saving locus data chunk at read {read_cursor+1}:')
            # log_memory_footprint()

        for pos, met in aln.locus_methylation.items():
            if met == '.':
                continue
            if include_unmethylated_ch or met.isupper() or (met.lower() == 'z'):
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

    return MethylationDataset(
        locus_data=locus_table, read_data=read_table
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




