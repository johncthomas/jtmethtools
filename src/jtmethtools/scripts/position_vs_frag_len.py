"""Plot the fragment position × fragment length methylation of read 1 & read 2.
Shows the effect of end repair, which in cell-free DNA can vary by fragment length (as a consequence
of fragment origin)."""
from typing import Self

import scipy.ndimage
from scipy.constants import sigma

import jtmethtools as jtm
from jtmethtools.util import plt_labels
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter, gaussian_filter, generic_filter
from scipy.interpolate import griddata
from pathlib import Path

from dataclasses import field, dataclass
import datargs


ARR_KEYS = (
    ('met1', 'tot1'),
    ('met1', 'tot1', 'met2', 'tot2'),
)

#type PosMet = dict[str, NDArray]



class PosMet:
    """Arrays of counts of total CpGs and methylated CpGs, in arrays by original position in
    reads × original fragment length. If single ended data, only met1 and tot1 are used, and
    true fragment length can't be calculated beyond the maximum length of the read.

    Overlapped flag indicates that the R1 and R2 arrays have been combined into one array."""
    def __init__(
            self,
            met1: NDArray,
            tot1: NDArray,
            met2: NDArray = None,
            tot2: NDArray = None,
            overlapped = False,
    ):
        self.met1 = met1
        self.tot1 = tot1
        self.met2 = met2
        self.tot2 = tot2

        m2 = met2 is not None
        t2 = tot2 is not None

        if m2 != t2:
            raise ValueError(f"met2 and tot2 must both be provided or both be None.")
        self.paired_end = m2

        self.overlapped = overlapped

    def smooth_filter_counts(self, min_depth, smooth, max_height=None) -> Self:
        """Applies uniform smoothing; enforces depth threshold; restores dictionary structure"""

        smooth = uniform_filter(self[:max_height], smooth)

        m = np.sum(smooth, axis=0) < min_depth
        smooth[:, m] = 0

        return smooth

    @property
    def mean1(self):
        """Mean methylation of read 1"""
        return self.met1/self.tot1
    @property
    def mean2(self):
        """Mean methylation of read 2"""
        if self.paired_end:
            return self.met2/self.tot2
        else:
            return None

    def overlap(self, width=300) -> Self:
        """Overlap the counts of read 1 and read 2, aligning reads to their position in the fragment.
        Returns a copy with paired_end=False and overlapped=True.
        """
        if not self.paired_end:
            raise ValueError("Must be paired end")
        t1, t2, m1, m2 = [self[t] for t in ('tot1', 'tot2', 'met1', 'met2')]
        tm = []
        for x1, x2 in [(t1, t2), (m1, m2)]:
            x1r = np.zeros((width, width), np.uint32)


            for i in range(width):
                x1r[0:i + 1, i] = x1[-i - 1:, i]
            x1r += x2
            tm.append(x1r)

        return PosMet(met1=tm[1], tot1=tm[0], overlapped=True)

    def to_csv(self, out_dir:Path|str, sample_name:str=''):
        """Write arrays to CSV files. File names will be prefixed with sample_name.
        out_dir will be created if it doesn't exist."""
        array_out_dir = Path(out_dir)
        array_out_dir.mkdir(parents=True, exist_ok=True)

        ks = ('met1', 'tot1', 'met2', 'tot2') if self.paired_end else ('met1', 'tot1')

        if sample_name:
            out_prefix = array_out_dir / f"{sample_name}."
        else:
            out_prefix = array_out_dir / ''
        for k in ks:
            if self.overlapped:
                k = k.replace('1', '_overlapped')
                k = k.replace('2', '_overlapped')
            np.savetxt(str(out_prefix) + f"read-pos-vs-frag-len.{k}.csv", self[k], delimiter=',', fmt='%d')

    @classmethod
    def from_csv(cls, in_dir:Path|str, sample_name:str=''):
        """Read arrays from CSV files."""
        array_in_dir = Path(in_dir)
        if sample_name:
            in_prefix = array_in_dir / f"{sample_name}."
        else:
            in_prefix = array_in_dir / ''

        arrays = {}
        for k in ('met1', 'tot1', 'met2', 'tot2'):
            arrays[k] = np.loadtxt(str(in_prefix) + f"read-pos-vs-frag-len.{k}.csv", delimiter=',', dtype=np.uint32)
        return cls(**arrays)



    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        keys = ['met1', 'tot1',]
        if self.paired_end:
            keys += ['met2', 'tot2',]
        for k in keys:
            yield (k, self[k])




def get_counts(bamfn,  paired_end=True, width=300, min_mapq=20) -> PosMet:
    """Get counts of read position (relative to fragment, not by reference) vs fragment length.
    Calculates mean methylation.

    Arrays are orientated such that the 3' ends of the original fragments are in the first row of the array."""
    arrays = {k: np.zeros((width, width), dtype=np.uint32)
                    for k in ARR_KEYS[paired_end]}
    counts_r1_r2 = PosMet(**arrays)

    for aln in jtm.iter_bam(bamfn, paired_end=paired_end):

        if aln.mapping_quality() < min_mapq:
            continue
        if aln.a2 is None:
            continue

        if aln.has_methylated_ch():
            continue

        frag_len = (aln.reference_end - aln.reference_start)

        if frag_len >= width:
            continue

        for a in (aln.a, aln.a2):
            if a is None:
                continue
            ri = ['2', '1'][a.is_read1]
            metstr = jtm.get_bismark_met_str(a)
            for q, r in a.get_aligned_pairs():

                if (q is None) or (r is None):
                    continue
                if metstr[q] in ('z', 'Z'):
                    frag_pos = q if a.is_forward else a.query_length - q
                    counts_r1_r2[f"met{ri}"][frag_pos, frag_len] += metstr[q] == 'Z'
                    counts_r1_r2[f"tot{ri}"][frag_pos, frag_len] += 1
    counts_r1_r2.tot1 = np.flipud(counts_r1_r2.tot1)
    counts_r1_r2.met1 = np.flipud(counts_r1_r2.met1)


    return counts_r1_r2


def smooth_counts(self, sigma=2) -> NDArray:
    """Apply gaussian smoothing with edge filling to avoid data loss at the edges."""

    for k, arr in self:
        def _nanmean_filter(values):
            vals = values[~np.isnan(values)]
            return vals.mean() if len(vals) else np.nan

        filled = arr.copy()

        mask = np.isnan(arr)

        local_means = generic_filter(
            arr,
            _nanmean_filter,
            size=20,
            mode='nearest'
        )

        filled[mask] = local_means[mask]
        smooth = scipy.ndimage.gaussian_filter(filled, sigma=sigma)
    return smooth


def plot_counts(podmet:PosMet, sigma=6, min_obs=20):

    import matplotlib as mpl
    with mpl.rc_context({
            'figure.dpi': 150,
            'figure.figsize': (5, 3.75),
            'savefig.bbox': 'tight',
            'hist.bins': 20,
            'ytick.labelsize': 'x-small',
            'xtick.labelsize': 'x-small',
            'axes.labelsize': 'small',

            # approximate seaborn whitegrid
            'axes.grid': True,
            'grid.linestyle': '-',
            'grid.alpha': 0.4,

            # approximate seaborn paper context
            'font.size': 10,
            'axes.titlesize': 11,
            'legend.fontsize': 9,
            'lines.linewidth': 1.2,
    }):
        figs = {}

        reads = (1, 2) if podmet.paired_end else (1,)

        for i in reads:
            if podmet.overlapped:
                read_label = "Overlapped reads"
            else:
                read_label = f"Read {i}"
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8),
                                           height_ratios=[0.6, .4])
            mt = smooth_counts(podmet[f'mean{i}'], sigma=sigma)
            mt[podmet[f"tot{i}"] < min_obs] = np.nan

            plt.sca(ax1)
            plt.imshow(
                mt,
                cmap='plasma', vmax=1, vmin=0.5
            )
            plt_labels("Fragment length", 'Read position (original strand)', f'Read{i}')
            plt.grid(True, alpha=0.4)
            plt.colorbar(shrink=0.3)

            plt.sca(ax2)

            plt.plot(
                np.nanmean(podmet[f'mean{i}'], axis=0)
            )
            plt_labels('Fragment length', 'Prop. methylation',
                       f"{read_label} methylation by fragment length")
            plt.tight_layout()

            rk = f"R{i}" if not podmet.overlapped else 'overlapped'

            figs[rk] = fig

    return figs


def main(bamfn, plot_out_dir, array_out_dir=None,
         sample_name=None, width=300, smooth=6, write_arrays=False,
         min_obs=20, plot_fmt='png'):
    """Run counts + plotting workflow and write outputs.

    Files are written with prefix: outdir / (sample_name + '.').
    """
    bamfn = Path(bamfn)
    plot_out_dir = Path(plot_out_dir)


    plot_out_dir.mkdir(parents=True, exist_ok=True)


    sample = sample_name or bamfn.stem

    counts_r1_r2 = get_counts(bamfn, width=width)

    figs = plot_counts(counts_r1_r2, sigma=smooth, min_obs=min_obs)
    counts_overlap = counts_r1_r2.overlap()
    figs |= plot_counts(counts_overlap, sigma=smooth, min_obs=min_obs)


    s = f"{sample}." if sample else ''
    plot_prefix = plot_out_dir / s
    for fig_k, fig in figs.items():
        fig.savefig(str(plot_prefix) + f"position-vs-frag-len.{fig_k}.{plot_fmt}", dpi=150, bbox_inches='tight')
        plt.close(fig)

    if write_arrays:
        counts_r1_r2.to_csv(array_out_dir, sample_name=sample)

    return counts_r1_r2


def cli():
    """Call main using CLIArgs"""
    args = datargs.parse(CLIArgs)

    if args.pe == args.se:
        raise ValueError("One of --pe or --se must be set to indicate paired or single end reads.")

    if args.plot_out_dir is None and args.array_out_dir is None:
        raise ValueError("Provide at least one of --plot-out-dir or --array-out-dir, otherwise no results are written.")

    plot_fmt = args.plot_fmt.lower().replace('.', '')

    main(
        bamfn=args.bamfn,
        plot_out_dir=args.plot_out_dir,
        array_out_dir=args.array_out_dir,
        sample_name=args.sample_name,
        width=args.width,
        smooth=args.smooth,
        write_arrays=args.array_out_dir is not None,
        min_obs=args.min_obs,
        plot_fmt=plot_fmt,
    )

@datargs.argsclass(
    description="""\
Plot fragment position × fragment length methylation for read 1 & read 2.

Shows the effect of end repair, which in cell-free DNA can vary by fragment
length (as a consequence of fragment origin). Produces one PNG per read
(read1 and read2) and overlapped reads showing the whole fragment in the plot output directory. 
Optionally writes the underlying count matrices as CSVs.

NOTE: overlapped reads assume equal amounts of trimming done to 5' and 3' ends
"""
)
class CLIArgs:
    """Command-line arguments for the position vs fragment length methylation plotting script."""
    bamfn: Path = field(metadata=dict(
        required=True,
        help="Bismark BAM file (paired-end, with methylation call string).",
        aliases=['-b'],
    ))
    plot_out_dir: Path = field(metadata=dict(
        help="Directory for output PNG plots. Created if it does not exist. By default, if a path is not provided, counts are not written.",
        aliases=['-o', '-p'],
    ), default=None)
    plot_fmt:str = field(default='png', metadata=dict(
        help="File format for output plots. Recommended options are 'png', 'pdf' or 'svg'. Default is 'png'.",
    ))
    array_out_dir: Path = field(metadata=dict(
        required=False,
        help=("Directory for output CSV count matrices (met1/tot1/met2/tot2). "
              "By default, if a path is not provided, counts are not written."),
        aliases=['-a'],
    ), default=None)
    sample_name: str = field(metadata=dict(
        help="Sample name used as the filename prefix for outputs. "
             "Defaults to the BAM file stem.",
        aliases=['-s'],
    ), default=None)
    width: int = field(metadata=dict(
        help="Maximum fragment length / read position to consider (bp). "
             "Fragments at or above this length are skipped, and the output "
             "count matrices are width × width.",
        aliases=['-w'],
    ), default=300)
    smooth: int = field(metadata=dict(
        help="Strength (sigma) of Gaussian smoothing of the array when plotting. Written arrays are not smoothed.",
    ), default=2)
    min_obs: int = field(metadata=dict(
        help="Minimum count threshold for a position to be plotted in the heatmap. Written arrays are not filtered."
    ), default=20)
    se: bool = field(default=False, metadata=dict(
        help="Set if sequencing is single-ends. Either --se or --pe must be set."
    ))
    pe: bool = field(default=False, metadata=dict(
        help="Set if sequencing is paired-ends. Either --pe or --se must be set."
    ))


if __name__ == "__main__":
    cli()