import typing
from pathlib import Path
from typing import Collection, Tuple, Self
import tarfile

import tempfile

import pandas as pd
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

import pysam
import json
from loguru import logger
logger.remove()

def set_logger(min_level='DEBUG'):

    logger.remove()
    if min_level in ('DEBUG', 'TRACE'):
        logger.add(print, level=min_level, filter=lambda record: record["level"].name in ('DEBUG', 'TRACE'))

    logger.add(lambda msg: print(f"\033[96m{msg}\033[0m"), level="INFO", format="{message}")




SplitTable = dict[str, pd.DataFrame]


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
    return genome



def load_bismark_calls_table(fn) -> pd.DataFrame:
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


def load_region_bed(fn) -> pd.DataFrame:
    regions = pd.read_csv(
        fn, sep='\t', header=None,
        dtype={0:str}
    )
    regions.columns = ['Chrm', 'Start', 'End', 'Name', ]

    regions.set_index('Name', inplace=True, drop=False)
    return regions


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

