import os

import jtmethtools as jtm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from loguru import logger
import dataclasses

@dataclasses.dataclass
class MethFreqResult:
    cpg_methylated: np.ndarray
    cpg_total: np.ndarray
    ch_methylated: np.ndarray
    ch_total: np.ndarray
    length:int

    def to_df(self):
        """Convert the result to a pandas DataFrame."""
        import pandas as pd
        data = {
            'Position': np.arange(self.length),
            'CpG_Methylated': self.cpg_methylated,
            'CpG_Total': self.cpg_total,
            'CpH_Methylated': self.ch_methylated,
            'CpH_Total': self.ch_total
        }
        return pd.DataFrame(data)

    def write_tsv(self, fn: Path | str):
        """Write the result to a TSV file."""
        df = self.to_df()
        df.to_csv(fn, index=False, sep='\t')
        logger.info(f"Methylation by position data written to {fn}")


def methylation_by_position(
        fn:Path, length=100, stopper=-1, paired_end=True
) -> MethFreqResult:
    """Calculate the fraction of methylation by position in the read, relative
    to the adapter.

    Parameters:
        fn : Path to the BAM file.
        length : Number of bases to consider from the start of the read.
        stopper : Stop after this many alignments, -1 means no limit.
        paired_end: Set to False if single-ended, to reduce memory usage.
    """
    logger.info(f"Calculating methylation by position for {fn} with length {length} and stopper {stopper}")
    bam_iterer = jtm.alignments.iter_bam_segments(fn, paired_end=paired_end)

    mcpg_arr = np.zeros(shape=100, dtype=np.uint32)
    cpg_arr = np.zeros(shape=100, dtype=np.uint32)
    mch_arr = np.zeros(shape=100, dtype=np.uint32)
    ch_arr = np.zeros(shape=100, dtype=np.uint32)

    ch_set = {'x', 'h', 'u', 'X', 'H', 'U'}

    start = datetime.now()
    for aln_i, (a1, a2) in enumerate(bam_iterer):
        if a1.is_unmapped or (paired_end and (not a1.is_proper_pair)):
            continue
        for a in (a1, a2):
            if a is None:
                continue
            # get the methylation string
            m = jtm.alignments.get_bismark_met_str(a)
            # reverse the read if it's not forward - metstr
            #   are always in forward orientation, so this puts the
            #   end next to the adapter
            if not a.is_forward:
                m = ''.join(m[::-1])
            for i, met in enumerate(m):
                if i > length - 1:
                    break
                # skip the most common one
                if met == '.':
                    continue
                if met.lower() == 'z':
                    cpg_arr[i] += 1
                    if met == 'Z':
                        mcpg_arr[i] += 1
                elif met in ch_set:
                    ch_arr[i] += 1
                    if met.isupper():
                        mch_arr[i] += 1

        if aln_i == stopper:
            break
    end = datetime.now()
    logger.info(f'Processed {aln_i+1} alignments in {end - start}.')

    return MethFreqResult(
        cpg_methylated=mcpg_arr,
        cpg_total=cpg_arr,
        ch_methylated=mch_arr,
        ch_total=ch_arr,
        length=length
    )

def plot_methylation_by_position(result:MethFreqResult,):
    """Plot the methylation by position."""
    clr1 = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
    clr2 = (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)
    x = np.arange(result.length)
    # fig, axes = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
    fig, ax1 = plt.subplots()
    ax1.plot(x, result.cpg_methylated / result.cpg_total,
             color=clr1)
    plt.ylabel('Frac CpG methylated', color=clr1)
    plt.xlabel('BP from adapter')
    ax1.tick_params(axis='y', labelcolor=clr1)
    ax2 = ax1.twinx()

    plt.plot(x, result.ch_methylated / result.ch_total,
             color=clr2)
    plt.ylabel('Frac CpH methylated', color=clr2)
    ax2.tick_params(axis='y', labelcolor=clr2)
    plt.grid()
    return fig


# command line interface with Datargs
import datargs
from dataclasses import field
@datargs.argsclass(description="""\
Get methylation fraction by position. Outputs a table and a figure to the out-dir, by default.
""")
class MethylationMetricsArgs:
    """Command line arguments for methylation metrics calculation."""
    bam: Path = field(metadata={
        'help': 'Path to the BAM file.',
        'aliases': ['-b'],
    }, )
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
    length: int = field(
        default=100,
        metadata={'help': 'Number of bases from start to include. (default: 100)'},
    )
    single_ended: bool = field(
        default=False,
        metadata={'help': 'Set if sequences are not paired end, otherwise it will use lots '
                          'of memory. (default: False)'},
    )
    head: int = field(
        default=-1,
        metadata={'help': 'Stop after this many alignments, -1 means no limit. (default: -1)'},
    )

    no_table: bool = field(
        default=False,
        metadata={'help': 'Do not write a table of the results. '}
    )
    no_fig: bool = field(
        default=False,
        metadata={'help': 'Do not write a figure of the results. '}
    )

    def validate(self):
        """Validate the arguments."""
        if self.no_table and self.no_fig:
            raise ValueError("At least one output must be allowed.")



def cli_met_by_pos():
    args = datargs.parse(MethylationMetricsArgs)
    args.validate()
    os.makedirs(args.out_dir, exist_ok=True)
    result = methylation_by_position(args.bam, length=args.length, stopper=args.head,
                                     paired_end=not args.single_ended)

    samp = args.sample_name or args.bam.stem
    prefix = Path(args.out_dir) / (samp + '.')

    if not args.no_table:
        result.write_tsv(str(prefix) + 'met_by_position.tsv')

    if not args.no_fig:
        fig = plot_methylation_by_position(result)
        fig.savefig(str(prefix)+'met_by_position.png', bbox_inches='tight', dpi=150)