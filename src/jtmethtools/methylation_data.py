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

from functools import lru_cache

import attrs

import numpy as np

from numpy.typing import NDArray

import pysam
from pysam import AlignedSegment, AlignmentFile

import pyarrow as pa
import pyarrow.compute as compute
from pyarrow import parquet

from attrs import define

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


MethylationReturnValues = collections.namedtuple(
    'MethylationReturnValues', ['locus_data', 'read_data']
)

def process_bam_methylation_data(
        bamfn:Path|str,
        first_i:int=0,
        last_i:int=np.inf,
        regions:Regions=None,
        paired_end=True,
        cannonical_chrm_only=True,
        include_unmethylated_ch=False,
        chunk_size=int(1e6),
) -> MethylationReturnValues:

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

    if filtering_by_region:
        def include_alignment(a) -> bool:
            return len(a.get_hit_regions(regions)) > 0
    else:
        if cannonical_chrm_only:
            def include_alignment(a) -> bool:
                return a.reference_name in CANNONICAL_CHRM
        else:
            def include_alignment(a) -> bool:
                return True

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

    return MethylationReturnValues(
        locus_data=locus_table, read_data=read_table
    )


def write_methylation_dataset(
        outdir:Path|str,
        locus_table:pa.Table,
        read_table:pa.Table,
        metadata:dict=None,
) -> None:
    """Write the methylation dataset to a directory."""
    outdir = Path(outdir)
    if not outdir.exists():
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




