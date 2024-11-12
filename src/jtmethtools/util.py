import typing
from pathlib import Path
from typing import Collection, Tuple, Self
import tarfile
import os
import json

import pandas as pd
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

import pysam

from loguru import logger
logger.remove()

def set_logger(min_level='DEBUG'):

    logger.remove()
    if min_level in ('DEBUG', 'TRACE'):
        logger.add(print, level=min_level, filter=lambda record: record["level"].name in ('DEBUG', 'TRACE'))

    logger.add(lambda msg: print(f"\033[96m{msg}\033[0m"), level="INFO", format="{message}")

set_logger()


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


def load_region_bed(fn) -> pd.DataFrame:
    regions = pd.read_csv(
        fn, sep='\t', header=None,
        dtype={0:str}
    )
    regions.columns = ['Chrm', 'Start', 'End', 'Name', ]

    regions.set_index('Name', inplace=True, drop=False)
    return regions


import json


def write_array(
        array: NDArray,
        outfile: str|Path,
        additional_metadata: dict = None
) -> None:
    """Write a tar file that contains the binary data and metadata
    to recreate the original numpy array.

    additional_metadata values should be strings or stringable."""
    # Create a temporary directory to store the files
    with tarfile.open(outfile, "w") as tar:
        # Save the binary data of the array
        data_filename = "data.bin"
        array.tofile(data_filename)
        tar.add(data_filename)

        # Create the metadata file with shape and dtype
        meta_filename = "metadata.json"

        metadata = {
            f'_np_shape': [s for s in array.shape],
            f'_np_dtype': str(array.dtype),
        }

        if additional_metadata is not None:
            metadata = metadata | additional_metadata

        with open(meta_filename, "w") as meta_file:
            json.dump(metadata, meta_file)
        tar.add(meta_filename)

    # Clean up temporary files
    os.remove(data_filename)
    os.remove(meta_filename)


def read_array(tar_filename: str|Path, ) -> Tuple[NDArray, dict]:
    """Read file created by `write_array`"""
    with tarfile.open(tar_filename, "r") as tar:
        # Extract the metadata file and parse it
        meta_file = tar.extractfile("metadata.json")

        metadata = json.load(meta_file)

        # Extract the binary data and load it as a NumPy array
        data_file = tar.extractfile("data.bin")

        array = np.frombuffer(
            data_file.read(),
            dtype=metadata['_np_dtype']
        ).reshape(metadata['_np_shape'])

    return array, metadata


