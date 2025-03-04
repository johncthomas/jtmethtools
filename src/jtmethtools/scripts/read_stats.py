from typing import Tuple, Iterator
from pathlib import Path
import  os
from collections import Counter
from dataclasses import field

import matplotlib.pyplot as plt
import pysam
import pandas as pd
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


def get_table(bamfn, only_cannonical_chrm=False) -> npt.NDArray[np.uint32]:
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


def plot_len_pmet(table:npt.NDArray) -> None:
    """Plot read length vs methylation status."""
    m = table[:, COLI['FreqC']] > 0
    x = table[m, COLI['Length']]
    y = table[m, COLI['Met']] / table[m, COLI['FreqC']]

    plt.figure(figsize=(4 ,4))
    plt.hexbin(x, y, bins='log', gridsize=50)
    plt.xlabel('Read Length (BP)')
    plt.ylabel('Methylation %')
    plt.colorbar(label='log10(count)')
    plt.title('Read Length vs Methylation %')



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
        write_array(
            table,
            str(out_prefix)+'.read-stats.arr.gz',
            additional_metadata={
                'columns': COLUMNS,
                'description': "Read stats from Bismark BAM.",
                'bamfn': str(bamfn),
            }
        )


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
        table, metadat = read_array(tmpdir/'test.read-stats.arr.gz')
        df = pd.DataFrame(table, columns=COLUMNS)
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
    print('Running test.')
    ttest()
    cli()