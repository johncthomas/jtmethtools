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

from jtmethtools.alignments import get_bismark_met_str
from jtmethtools.util import write_array, read_array


COLUMNS = ( 'Chrm', 'Start', 'Length', 'H', 'X', 'U', 'Z', 'h', 'x', 'u', 'z', 'FreqC', 'Met', 'NotC')
COLI = {c:i for i, c in enumerate(COLUMNS)}

CHRM_MAP:dict[str, int] = {}
for i, c in enumerate(list(range(1, 24))+['X', 'Y']):
    CHRM_MAP[str(c)] = i
    CHRM_MAP[f"chr{c}"] = i

VALID_CHRM = set(list(CHRM_MAP.keys()))


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


def get_table_np(bamfn, only_cannonical_chrm=False) -> npt.NDArray[np.uint32]:
    """Generate table of per-read methylation stats from a Bismark BAM."""
    i = 0
    for i, _ in enumerate(iter_alignments(bamfn, only_cannonical_chrm)):
        pass
    total_alignments = i+1

    # create arrays to hold the data
    table = np.zeros((total_alignments, len(COLUMNS)), dtype=np.uint32)

    for i, a in enumerate(iter_alignments(bamfn, only_cannonical_chrm)):
        ln = a.query_length

        metstr = get_bismark_met_str(a)

        row = np.zeros(len(COLUMNS), dtype=np.uint32)
        row[COLI['Chrm']] = CHRM_MAP[a.reference_name]
        row[COLI['Start']] = a.query_alignment_start
        row[COLI['Length']] = ln

        # count glyphs in the methylation string.
        c = dict(Counter(metstr))
        # Give '.' a better name
        c['NotC'] = c['.']
        del c['.']
        for k, v in c.items():
            row[COLI[k]] = v

        table[i] = row

    c_cols = [COLI[k] for k in 'H X U Z h x u z'.split()]
    met_cols = [COLI[k] for k in 'H X Z U'.split()]
    table[:, COLI['FreqC']] = table[:, c_cols].sum(1)
    table[:, COLI['Met']] = table[:, met_cols].sum(1)

    return table


def get_table(bamfn, only_cannonical_chrm=False) -> pd.DataFrame:
    """Generate table of per-read methylation stats from a Bismark BAM."""
    i = 0
    for i, _ in enumerate(iter_alignments(bamfn, only_cannonical_chrm)):
        pass
    total_alignments = i+1
    # Preallocate a dictionary of numpy arrays (one per column).
    # Adjust the list of columns as needed.
    columns = ['Chrm', 'Start', 'Length', 'H', 'X', 'U', 'Z',
               'h', 'x', 'u', 'z', 'FreqC', 'Met', 'NotC']
    data_dict = {col: np.zeros(total_alignments, dtype=np.uint32) for col in columns}

    # Second pass: populate arrays for each alignment
    for i, a in enumerate(iter_alignments(bamfn, only_cannonical_chrm)):
        ln = a.query_length
        metstr = get_bismark_met_str(a)

        # Static fields
        data_dict['Chrm'][i] = CHRM_MAP[a.reference_name]
        data_dict['Start'][i] = a.query_alignment_start
        data_dict['Length'][i] = ln

        # Count glyphs in the methylation string.
        c = Counter(metstr)
        # Rename '.' to 'NotC'
        c['NotC'] = c.get('.', 0)
        if '.' in c:
            del c['.']

        # For each key from the counter that matches one of our columns, set the value.
        for key, v in c.items():
            if key in data_dict:
                data_dict[key][i] = v
        # (Any columns not updated remain at 0.)

    # Compute derived columns FreqC and Met.
    # Define which keys to sum over for each derived column.
    freqc_keys = ['H', 'X', 'U', 'Z', 'h', 'x', 'u', 'z']
    met_keys = ['H', 'X', 'Z', 'U']

    # Sum across the selected keys, vectorized.
    data_dict['FreqC'] = sum(data_dict[key] for key in freqc_keys)
    data_dict['Met'] = sum(data_dict[key] for key in met_keys)

    # Convert each numpy array to a pyarrow array.
    # Zero-copy is achieved for many numeric types.
    arrow_dict = {col: pa.array(arr) for col, arr in data_dict.items()}

    # convert to arrow backed DF.
    table = pa.table(arrow_dict)
    df = table.to_pandas(types_mapper=pd.ArrowDtype, self_destruct=True)

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

    table = get_table(bamfn, cannon_chrm)

    if do_plot:
        plot_len_pmet(table)

        plt.savefig(
            str(out_prefix)+'.len-vs-met.png',
            dpi=300,
            bbox_inches='tight'
        )

    if write_table:
        table.to_parquet(
            str(out_prefix)+'.read-stats.parquet',
        )
        # write_array(
        #     table,
        #     str(out_prefix)+'.read-stats.arr.gz',
        #     additional_metadata={
        #         'columns': COLUMNS,
        #         'description': "Read stats from Bismark BAM.",
        #         'bamfn': str(bamfn),
        #     }
        # )


def ttest():
    from tempfile import TemporaryDirectory
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
        df = pd.read_parquet(tmpdir/'test.read-stats.parquet')

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
    out_prefix: Path = field(metadata=dict(
        required=True,
        help="Output files will begin with this. If output dir doesn't exist, it will be created.",
        aliases=["-o"]
    ))
    cannonical_chrm: bool = field(metadata=dict(
        default=False,
        help='Only include reads from cannonical chromosomes.'
    ))
    no_plot: bool = field(metadata=dict(
        default=False,
        help='Do not generate a plot of length vs methylation status.'
    ))
    no_table: bool = field(metadata=dict(
        default=False,
        help='Do not write table of read stats.'
    ))


def cli():
    args = datargs.parse(ReadStatsArgs)
    run(
        args.bamfn,
        args.out_prefix,
        not args.no_plot,
        not args.no_table,
        args.cannonical_chrm
    )

if __name__ == '__main__':
    # print('Running test.')
    # ttest()
    cli()