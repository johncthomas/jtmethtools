import sys
from typing import Tuple, Iterator
from pathlib import Path
import  os
from collections import Counter
from dataclasses import field

import matplotlib.pyplot as plt
import pysam
import pandas as pd
import pyarrow as pa
import numpy as np
import numpy.typing as npt
import datargs
from loguru import logger

from jtmethtools.alignments import get_bismark_met_str
from jtmethtools.util import write_array, read_array





COLUMNS = ( 'Chrm', 'Start', 'Length', 'H', 'X', 'U', 'Z', 'h', 'x', 'u', 'z', 'NotC', 'FreqC', 'Met', "Read")
COLI = {c:i for i, c in enumerate(COLUMNS)}

CHRM_MAP:dict[str, int] = {}
for i, c in enumerate(list(range(1, 24))+['X', 'Y']):
    CHRM_MAP[str(c)] = i
    CHRM_MAP[f"chr{c}"] = i

VALID_CHRM = set(list(CHRM_MAP.keys()))


def table2df(table: pa.Table) -> pd.DataFrame:
    """Convert a pyarrow Table to a pandas DataFrame, converting dictionary types to pandas categorical types."""
    mapping = {schema.type: pd.ArrowDtype(schema.type) for schema in table.schema}

    return table.to_pandas(types_mapper=mapping.get, ignore_metadata=True)


def read_parquet(fn) -> pd.DataFrame:
    """load a parquet file and convert to a pandas dataframe.
    Works when pd.read_parquet fails on the dictionary types."""
    tls = pa.parquet.read_table(fn)
    tls = table2df(tls)
    return tls

# Create mappings for dictionary encodings
def mapping_to_pa_dict(mapping: dict[str, int], values: npt.NDArray, ) -> pa.DictionaryArray:
    """Take a encoded numpy array and the dictionary that decodes
    it, return the dictionary encoded arrow array."""
    # get values in the order of the mapping so they'll be the same in the
    #  final dictionary
    i2s = {v: k for k, v in mapping.items()}
    dict_values = [i2s[i] for i in range(max(i2s.keys()) + 1)]
    dictionary = pa.array(dict_values, type=pa.string())

    # Convert to dict array. Pandas needs 32bit array to convert to
    indices = pa.array(values, type=pa.int32())
    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    return dict_array

def iter_alignments(bamfn, only_cannon_chrm) -> Iterator[pysam.AlignedSegment]:
    for i, a in enumerate(pysam.AlignmentFile(bamfn)):
        if not a.is_mapped:
            continue
        chrm = a.reference_name
        if only_cannon_chrm:
            if chrm not in VALID_CHRM:
                continue
            yield a
        else:
            try:
                chrm_i = CHRM_MAP[chrm]
            except KeyError:
                chrm_i = max(CHRM_MAP.values())+1
                CHRM_MAP[chrm] = chrm_i
            yield a


def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    return mem


def _get_stats_array(bamfn, only_cannonical_chrm=False) -> npt.NDArray[np.uint32]:
    """Generate table of per-read methylation stats from a Bismark BAM."""
    logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")
    i = 0
    for i, _ in enumerate(iter_alignments(bamfn, only_cannonical_chrm)):
        pass
    total_alignments = i + 1
    logger.info(f"Total alignments: {total_alignments}")

    # create arrays to hold the data
    table = np.zeros((total_alignments, len(COLUMNS)), dtype=np.uint32)
    logger.info('Created empty table, memory usage: {:.2f} MB'.format( get_memory_usage() ))
    chrm_i = COLI['Chrm']
    start_i = COLI['Start']
    length_i = COLI['Length']
    read_i = COLI['Read']
    for i, a in enumerate(iter_alignments(bamfn, only_cannonical_chrm)):
        ln = a.query_length

        metstr = get_bismark_met_str(a)

        #row = table[i]
        table[i, chrm_i] = CHRM_MAP[a.reference_name]
        table[i, start_i] = a.query_alignment_start
        table[i, length_i] = ln
        table[i, read_i] = 1 if a.is_read1 else 2


        # count glyphs in the methylation string.
        c = dict(Counter(metstr))
        logger.debug(metstr)
        logger.debug(c)
        # Give '.' a better name
        c['NotC'] = c['.']
        del c['.']
        for k, v in c.items():
            table[i, COLI[k]] = v

    c_cols = [COLI[k] for k in 'H X U Z h x u z'.split()]
    met_cols = [COLI[k] for k in 'H X Z U'.split()]
    table[:, COLI['FreqC']] = table[:, c_cols].sum(1)
    table[:, COLI['Met']] = table[:, met_cols].sum(1)

    logger.info('Populated table, memory usage: {:.2f} MB'.format(get_memory_usage()))

    return table


def _array_to_df(table: npt.NDArray[np.uint32]) -> pd.DataFrame:
    """Convert the numpy array of read stats to an Arrow backed DataFrame."""
    cols_not_chrm = [col for col in COLUMNS if col != 'Chrm']
    arrow_dict = {'Chrm': mapping_to_pa_dict(CHRM_MAP, table[:, COLI['Chrm']])}
    arrow_dict |= {col: pa.array(table[:, COLI[col]]) for col in cols_not_chrm}

    table = pa.table(arrow_dict)
    df = table.to_pandas(types_mapper=pd.ArrowDtype, self_destruct=True)
    return df

def get_stats_df(bamfn, only_cannonical_chrm=False) -> pd.DataFrame:
    """Generate table of per-read methylation stats from a Bismark BAM."""
    arr = _get_stats_array(bamfn, only_cannonical_chrm)
    df = _array_to_df(arr)
    return df


def plot_len_pmet(readstats:pd.DataFrame, exclude_CH=False,
                  legend=True, hexbin=True) -> None:
    """Plot read length vs methylation status."""
    readstats.loc[:, 'MetFraction'] = readstats.Met / readstats.FreqC
    if exclude_CH:
        ch_met = (readstats.H > 0) | (readstats.X > 0) | (readstats.U > 0)
        rs = readstats.loc[~ch_met]
    else:
        rs = readstats

    x, y = rs.Length.values, rs.MetFraction.values * 100
    plt.figure(figsize=(4, 4))
    if hexbin:
        plt.hexbin(x, y, bins='log', gridsize=25)
        lc='lightblue'
    else:
        lc = 'black'
    plt.xlabel('Read Length (BP)')
    plt.ylabel('Methylation %')
    # plt.colorbar(label='log(count)')
    plt.title('Read Length vs Methylation %')
    chunk_w = 10

    for q, ls in ((0.5, ':'), (0.75, '--'), (0.95, '-')):
        qx, qy = [], []
        for chunk_n in range(150 // chunk_w + 1):
            # we're ignore reads of length 1 in exchange for putting length 141-150 into one bucket
            start = chunk_n * chunk_w
            end = (chunk_n + 1) * chunk_w
            m = (rs.Length > start) & (rs.Length <= end)

            med = rs.loc[m, 'MetFraction'].quantile(q)
            qx.append(end)
            qy.append(med)
        qy = [y * 100 if not pd.isna(y) else 0 for y in qy]
        plt.plot(qx, qy, label=str(q), c=lc, ls=ls, lw=3)
    if legend:
        plt.legend(title='Quantile', fontsize='xx-small', loc='upper left', bbox_to_anchor=(1, 1))


def run(bamfn, out_prefix, do_plot, write_table, cannon_chrm):

    out_prefix = Path(out_prefix)
    if not os.path.isdir(out_prefix.parent):
        os.makedirs(out_prefix.parent)

    table = get_stats_df(bamfn, only_cannonical_chrm=cannon_chrm)
    logger.info(f"Generated stats table, memory usage: {get_memory_usage():.2f} MB")
    if write_table:
        table.to_parquet(
            str(out_prefix)+'.read-stats.parquet',
        )
    if do_plot:
        plot_len_pmet(table)

        plt.savefig(
            str(out_prefix)+'.len-vs-met.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()



def ttest():
    from tempfile import TemporaryDirectory
    logger.remove()
    logger.add(print, level="DEBUG", colorize=True)

    test_sam = """\
@HD	VN:1.0	SO:none
@SQ	SN:allCpG	LN:10
n1	2	allCpG	1	42	10M	*	0	10	NNNNNNNNNN	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:UHZXuhzx..
n2	2	allCpG	1	42	5M	*	0	10	NNNNN	ABCDE	NM:i:tag0\tMD:Z:tag1\tXM:Z:UHZX.
n3	2	allCpG	1	42	10M	*	0	10	NNNNNNNNNN	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
"""
    samfn = 'test.sam'
    # write test_sam to tempdir
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        samfn =  tmpdir/samfn
        with open(samfn, 'w') as f:
            f.write(test_sam)
        # run main
        run(samfn, tmpdir / 'test', True, True, cannon_chrm=False)

        # read with pyarrow
        df = read_parquet(tmpdir/'test.read-stats.parquet')
        print(df)
        # check some numbers in df
        assert df.loc[0, 'FreqC'] == 8
        assert df.loc[0, 'Met'] == 4
        assert df.loc[1, 'FreqC'] == 4
        assert df.loc[1, 'Met'] == 4
        assert df.loc[2, 'Length'] == 10
        assert df.loc[2, 'NotC'] == 10

    print("Tests finished.")





@datargs.argsclass(
    description="""\
Generate table of per-read methylation stats, and plot length vs methylation status, from a 
Bismark BAM.

Output table counts methylation type, context and other read stats. Uses Bismark letter 
coding for methylation, except with "NotC" column giving counts of non-C nucleotides instead of ".".
"""
)
class ReadStatsArgs:
    bamfn: Path = field(metadata=dict(
        required=True,
        help="Bismark BAM file (only one).",
        aliases=['-b']
    ))
    out_dir: Path = field(metadata=dict(
        required=True,
        help="Output files will begin with this. If output dir doesn't exist, it will be created.",
        aliases=["-o"]
    ))
    sample_name: str = field(metadata=dict(
        default=None,
        help="Sample name to use in output files. Default is taken from the bam name.",
        aliases=["-s"]
    ))
    all_chrm: bool = field(metadata=dict(
        default=False,
        help='By default, only include reads from cannonical chromosomes. Set this to include all chromosomes.'
    ))
    # plot_len_vs_m: bool = field(metadata=dict(
    #     default=False,
    #     help='Generate a plot of read length vs '
    # ))
    # no_table: bool = field(metadata=dict(
    #     default=False,
    #     help='Do not write table of read stats.'
    # ))
    quiet: bool = field(metadata=dict(
        default=False,
        help="Don't print logging messages."
    ))


def read_stats_cli(args:ReadStatsArgs=None):
    """Command line interface for read stats."""
    if args is None:
        args = datargs.parse(ReadStatsArgs)
    if args.sample_name is None:
        samp = args.bamfn.stem
    else:
        samp = args.sample_name
    logger.remove()
    if not args.quiet:
        logger.add(sys.stdout, level="INFO", colorize=True)

    out_prefix = Path(args.out_dir) / samp

    run(
        args.bamfn,
        out_prefix,
        # do_plot=args.plot_len_vs_m,
        # write_table=not args.no_table,
        do_plot=False,
        write_table=True,
        cannon_chrm=not args.all_chrm
    )
#
# logger.add(sys.stdout, level="INFO", colorize=True)
# tbl = _get_stats_array('/media/jcthomas/JCT61-1/Data/250330_3datasets/CMDL19003184.deduplicated.sorted.bam')
# table = (tbl, '/home/jcthomas/test_read_stats.parquet')
#ttest_np()
if __name__ == '__main__':
    import sys
    if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
        ttest()
    else:
        read_stats_cli()