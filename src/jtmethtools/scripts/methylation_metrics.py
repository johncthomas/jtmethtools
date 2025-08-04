import collections
import os
from collections import Counter
from typing import Callable, Tuple, Iterator
import json
import jtmethtools as jtm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import pysam
from loguru import logger
import dataclasses
from scipy import stats, signal
from jtmethtools import Regions
from jtmethtools.alignments import Alignment, alignment_overlaps_region



@dataclasses.dataclass
class MethFreqResult:
    cpg_methylated: np.ndarray
    cpg_total: np.ndarray
    ch_methylated: np.ndarray
    ch_total: np.ndarray
    length:int

    def to_df(self):
        """Convert the result to a pandas DataFrame."""
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

    mcpg_arr = np.zeros(shape=length, dtype=np.uint32)
    cpg_arr = np.zeros(shape=length, dtype=np.uint32)
    mch_arr = np.zeros(shape=length, dtype=np.uint32)
    ch_arr = np.zeros(shape=length, dtype=np.uint32)

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
class ArgsPosMet:
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


def cli_met_by_pos(args=None):
    if args is None:
        args = datargs.parse(ArgsPosMet)
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


def iter_filtered_se_v2(
        bamfn: Path,
        regions: Regions,
        counter: Counter,
        min_mapq: int = 0,
        min_ncpg: int = 0,
) -> Iterator[pysam.AlignedSegment]:
    bam_iterer = jtm.alignments.iter_bam_segments(
        bamfn, paired_end=False
    )


    for alns in bam_iterer:
        for aln in alns:

            if aln is None:
                continue

            counter['TotalReads'] += 1

            if aln.is_unmapped:
                counter['UnmappedReads'] += 1
                continue

            if not alignment_overlaps_region(aln, regions):
                counter['MissesRegions'] += 1
                continue

            counter['AlignedHits'] += 1

            metstr = jtm.alignments.get_bismark_met_str(aln)
            low_metstr = metstr.lower()

            if min_mapq and aln.mapping_quality < min_mapq:
                counter['LowMapQ'] += 1
                continue

            if min_ncpg and low_metstr.count('z') < min_ncpg:
                counter['LowCpGs'] += 1
                continue

            yield aln


#
# def met_stats_of_regions(
#         ba m,
#         regions: Regions,
#         min_mapq: int = 0,
#         min_ncpg: int = 0,
# ):
#     """Calculate coverage table from filtered BAM file.
#
#     Coverage table is a per locus count of CpG and CpH methylation."""
#


def met_stats_of_regions_v1(
        bamfn: Path,
        regions: Regions,
        min_mapq: int = 0,
        min_ncpg: int = 0,
        get_loci_methylation=True
) -> tuple[Counter, dict[Tuple[str, int], list[int]]]:
    """Calculate methylation statistics of a filtered BAM file.

    Number/proportion of aligned reads, and number of methylated CpG and CpH.

    Probably makes sense to treat everything as single ended.

    """
    results = Counter()
    if get_loci_methylation:
        methylation_counts:dict[Tuple[str, int], list[int]] = collections.defaultdict(lambda : [0, 0]) # (methylated, total)
    else:
        methylation_counts = None

    for aln in iter_filtered_se_v2(
            bamfn, counter=results, regions=regions, min_mapq=min_mapq, min_ncpg=min_ncpg
    ):
        metstr = jtm.alignments.get_bismark_met_str(aln)
        has_methylated_ch = any(c in {'X', 'H', 'U'} for c in metstr)

        for q_pos, r_pos in aln.get_aligned_pairs(matches_only=True):

            met = metstr[q_pos]
            low_met = met.lower()

            if low_met == 'z':
                results['TotalCpG'] += 1
                if met == 'Z':
                    results['MethylatedCpG'] += 1

                if not has_methylated_ch:
                    results['TotalCpG_NoCpH'] += 1
                    if met == 'Z':
                        results['MethylatedCpG_NoCpH'] += 1

                if get_loci_methylation and (not has_methylated_ch):
                    vals = methylation_counts[(aln.reference_name, r_pos)]
                    if (met == 'Z'):
                        vals[0] += 1
                    else:
                        vals[1] += 1

            elif low_met in {'x', 'h', 'u'}:
                results['TotalCpH'] += 1
                if met.isupper():
                    results['MethylatedCpH'] += 1
    return results, methylation_counts


import argparse
from pathlib import Path

def args_met_stats_in_regions():
    description = """\
Get methylation stats from regions given by BED file. Outputs a
JSON file with the stats for each given BAM file.

Output keys (presence may vary):
    TotalReads: Total number of reads in the BAM, aligned or not.
    AlignedHits: Number of aligned reads that overlap with the 
        regions. All CpG and CpH stats calculated using these reads.
    TotalCpG: Total number of CpGs in the aligned reads.
    MethylatedCpG: Number of methylated CpGs in the aligned reads.
    TotalCpG_NoCpH: Total number of CpGs in the aligned reads, after 
        removing reads with CpH methylation.
    MethylatedCpG_NoCpH: Number of methylated CpGs in the aligned 
        reads, after removing reads with CpH methylation.
    TotalCpH: Total number of CpHs in the aligned reads.
    MethylatedCpH: Number of methylated CpHs in the aligned reads.
    UnmappedReads: Number of reads that are not mapped.
    MissesRegions: Number of aligned reads that do not overlap with 
        the regions.
    LowMapQ: Number of aligned reads with mapping quality below the 
        minimum.
    LowCpGs: Number of aligned reads with less than the minimum 
        number of methylated CpGs.
    ModeLow: Mode of beta <= 0.6 
    ModeHigh: Mode of beta > 0.6
    

Presence/absence of key depends on options. E.g. if you're not 
filtering by mapping quality, the LowMapQ key will not be present.
"""

    parser = argparse.ArgumentParser(
        description=description,
        # raw description keeps the newlines.
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-b', '--bam',
                        required=True,
                        type=Path,
                        help='Path to BAM file.')
    parser.add_argument('-r', '--regions',
                        required=True,
                        type=Path,
                        help='BED or TSV file with regions to calculate stats in.')
    parser.add_argument('-o', '--out-dir',
                        required=True,
                        type=Path,
                        help='Output directory to write the results to.')
    parser.add_argument('-s', '--sample-name',
                        type=str,
                        default=None,
                        help='Sample name to use in output files. '
                             'Default is taken from the BAM name.')
    parser.add_argument('--min-mapq',
                        type=int,
                        default=0,
                        help='Minimum mapping quality to consider a read. (default: 0)')
    parser.add_argument('--min-ncpg',
                        type=int,
                        default=0,
                        help='Minimum number of methylated CpGs to consider a read. '
                             '(default: 0)')
    parser.add_argument('--beta-plots',
                        action='store_true',
                        help='Plot the distribution of beta values and modes. '
                             'Plots saved to $outdir/beta_plots/ with the BAM name as prefix. ')
    parser.add_argument('--no-modes',
                        action='store_true',
                        help='Set to not calculate modes of beta values, speeding up operation.')
    parser.add_argument('--min-depth',
                        type=int,
                        default=(dflt_depth:=50),
                        help='Minimum depth to include a CpG in beta density calculation. '
                             f'(default: {dflt_depth})'),
    parser.add_argument('--record-density',
                        action='store_true',
                        help='Record the density of methylation betas. '
                             'Will be saved in the JSON as "DensityX" and "DensityY".')
    parser.add_argument('-q', '--quiet',
                        action='store_true',
                        help='Do not print progress messages to stdout.')

    return parser


def methylation_beta_density(
        betas:pd.Series,
        adjust: float = 1.0,
        grid_n: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    # Clean & clip
    vals = betas.dropna().to_numpy()
    vals = vals[(vals >= 0) & (vals <= 1)]
    if vals.size == 0:
        return np.array([]), np.array([])

    # Kernel density estimate
    kde = stats.gaussian_kde(vals, bw_method='scott')
    kde.set_bandwidth(kde.factor * adjust)

    x = np.linspace(0, 1, grid_n)
    y = kde(x)
    return x, y


def lower_upper_mode(
        x:np.ndarray, y:np.ndarray,
        right_threshold: float = 0.6,
) -> Tuple[float, float]:
    """
    Returns (left_mode, right_mode) for a 0-1 bounded series.
    args:
        adjust: bandwidth multiplier (≈ R’s `adjust`)
        grid_n: points in the KDE grid
        right_threshold: x-value above which to look for the “upper” mode
        do_plot: if True, make a plot
    """

    # Local maxima (vectorised & fast)
    peaks, _ = signal.find_peaks(y)
    if peaks.size == 0:
        logger.warning("No modes found.")
        return None, None

    mode_x = x[peaks]
    mode_y = y[peaks]

    left_mode = mode_x[np.argmin(mode_x)]  # closest to 0
    right_candidates = mode_x[mode_x > right_threshold]  # > 0.6 like the R code
    right_mode = (
        right_candidates[np.argmax(mode_y[mode_x > right_threshold])]
        if right_candidates.size else None
    )

    return left_mode, right_mode


def plot_beta_distribution_modes(
        x:np.ndarray,  y:np.ndarray,
        lower_mode:float, upper_mode:float
) -> None:

    plt.plot(x, y)
    ymin, ymax = plt.ylim()
    plt.plot([lower_mode, lower_mode], [ymin, ymax], 'k--', lw=0.75, alpha=0.7)
    plt.plot([upper_mode, upper_mode], [ymin, ymax], 'k--', lw=0.75, alpha=0.7)
    # stop it expanding the limits
    plt.ylim(ymin, ymax)
    plt.xlabel('Beta value')
    plt.ylabel('Density')

    # textbox with modes
    textstr = f'Lower mode: {lower_mode:.3f}\nUpper mode: {upper_mode:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(
        0.5, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=props
    )


def cli_met_stats_in_regions(args=None):
    import importlib.metadata
    logger.info(
        f"Version: {importlib.metadata.version('jtmethtools')}"
    )

    if args is None:
        parser = args_met_stats_in_regions()
        args = parser.parse_args()

    if args.regions.suffix == '.bed':
        regions = Regions.from_bed(args.regions)
    elif (args.regions.suffix == '.tsv') or (args.regions.suffix == '.txt'):
        regions = Regions.from_file(args.regions)
    else:
        raise ValueError(f"Unsupported regions file format: {args.regions.suffix}. "
                         "Only .bed or tab separated tables with .tsv|.txt are supported.")

    bamfn = Path(args.bam)
    sample_name = args.sample_name or bamfn.stem

    import sys
    logger.add(sys.stdout, level='INFO', colorize=True)
    if not args.quiet:
        logdir = args.out_dir/'log'
        logdir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        logger.add(logdir/f'{sample_name}.{timestamp}.log', level='INFO', )

    # print args
    logger.info(f"Arguments: {args}")

    # create dir if requred
    out_dir = args.out_dir

    if not out_dir.exists():
        logger.info(f"Creating output directory {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
    if args.beta_plots is not None:
        (out_dir/'beta_plots').mkdir(parents=True, exist_ok=True)

    get_modes = not args.no_modes

    logger.info(f"Calculating methylation stats for sample {sample_name} in {args.regions}")

    # this is currently split in a weird way, first half is in the following function
    #   call, but then a bunch is done after the call. This isn't ideal, but
    #   I'm not going to refactor it right now.
    res, loci_methylation = met_stats_of_regions_v1(
        bamfn=bamfn,
        regions=regions,
        min_mapq=args.min_mapq,
        min_ncpg=args.min_ncpg,
        get_loci_methylation=get_modes
    )
    res = dict(res)
    res['BAM'] = str(bamfn)
    res['Regions'] = str(args.regions)

    if get_modes or args.record_density:
        logger.debug("Calculated betas and density x,y")
        betas = pd.DataFrame(
            loci_methylation.values(),
            columns=['Methylated', 'Unmethylated']
        )
        betas['Obs'] = (betas['Methylated'] + betas['Unmethylated'])
        # filter betas by minimum depth
        betas = betas.loc[betas.Obs >= args.min_depth]
        betas['Beta'] = betas['Methylated'] / betas.Obs
        density_x, density_y = methylation_beta_density(
            betas['Beta'],
        )
        if args.record_density:
            res['DensityX'] = density_x.tolist()
            res['DensityY'] = density_y.tolist()

        if get_modes:
            logger.debug("Getting modes")

            low_mode, hi_mode = lower_upper_mode(
                density_x, density_y,
            )
            res['ModeLow'], res['ModeHigh'] = low_mode, hi_mode
            if args.beta_plots:
                logger.info("Plotting beta distribution, modes.")
                plot_beta_distribution_modes(density_x, density_y,
                                             low_mode, hi_mode)
                plt.title(f"{sample_name}\n{args.regions.stem}")

                plt.savefig(
                    out_dir/'beta_plots'/f"{sample_name}.beta-distribution.png",
                    bbox_inches='tight', dpi=150
                )
                plt.close()
        else:
            logger.debug("Not getting modes.")

    # write res as a JSON
    out_json = out_dir / f"{sample_name}.methylation_stats.json"
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=4)

    logger.info(f"Results written to {args.out_dir}")


def ttest_statsin_regions():
    home = str(Path.home())
    bamfn = f'{home}/hdc-bigdata/data/Canary/bam/250301_twist-canary/sorted_coords/CMDL19003184.deduplicated.sorted.bam'
    regfn = '~/DevLab/NIMBUS/Reference/dmrs_extended_full_annotation_20200117.bed'
    outd = '~/tmp/250725.region-methylation'
    argtokens = f"-b {bamfn} -r {regfn} -s sample -o {outd} --beta-plots".replace(
        '~', home
    ).split()
    #args = datargs.parse(ArgsStatsInRegions, argtokens)
    parser = args_met_stats_in_regions()
    cli_met_stats_in_regions(parser.parse_args(argtokens))

if __name__ == '__main__':
    ttest_statsin_regions()
    #ttest_statsin_regions()
    #cli_met_stats_in_regions()