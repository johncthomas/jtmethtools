import matplotlib.pyplot as plt
import pysam
from collections import Counter
from jtmethtools.alignments import get_bismark_met_str
import  os
import pandas as pd
from pathlib import Path
import datargs
from dataclasses import field

def get_table(bamfn) -> pd.DataFrame:
    """Generate table of per-read methylation stats from a Bismark BAM."""
    table = []
    for i, a in enumerate(pysam.AlignmentFile(bamfn)):
        ln = a.query_length

        metstr = get_bismark_met_str(a)

        row = {'AlignmentIndex' :i, 'Chrm' :a.reference_name, 'Start' :a.query_alignment_start, 'Length' :ln}

        # count glyphs in the methylation string.
        c = dict(Counter(metstr))
        # change "." to "H"
        c['NotC'] = c['.']
        del c['.']
        row = row | c

        table.append(row)
    table = pd.DataFrame(table)
    table = table.fillna(0).astype(int, errors='ignore', copy=False)

    table.loc[:, 'FreqC'] = table.reindex(columns=['H', 'X', 'U', 'Z', 'h', 'x', 'u', 'z' ,]).sum(1).astype(int)
    table.loc[:, 'Met'] = table.reindex(columns=['H', 'X', 'Z', 'U']).sum(1).astype(int)

    return table


def plot_len_pmet(table) -> None:
    """Plot read length vs methylation status."""
    m = table.FreqC > 0
    x = table.loc[m, 'Length']
    y = table.loc[m, 'Met'] / table.loc[m, 'FreqC']
    plt.figure(figsize=(4 ,4))
    plt.hexbin(x, y, bins='log', gridsize=50)
    plt.xlabel('Read Length (BP)')
    plt.ylabel('Methylation %')


def run(bamfn, out_prefix, do_plot, write_table):

    out_prefix = Path(out_prefix)
    if not os.path.isdir(out_prefix.parent):
        os.makedirs(out_prefix.parent)

    table = get_table(bamfn)

    if do_plot:
        plot_len_pmet(table)
        plt.savefig(str(out_prefix)+'.len-vs-met.png')
    if write_table:
        table.to_csv(
            str(out_prefix)+'.read-stats.tsv.gz',
            sep='\t',
            index=False,
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
        run(samfn, tmpdir / 'test', True, True)
        df = pd.read_csv(tmpdir/'test.read-stats.tsv.gz', sep='\t', index_col=0)
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
    run(args.bamfn, args.out_prefix, not args.no_plot, not args.no_table)

if __name__ == '__main__':
    print('Running test.')
    ttest()
    cli()