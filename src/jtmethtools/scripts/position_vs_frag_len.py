"""Plot the fragment position × fragment length methylation of read 1 & read 2.
Shows the effect of end repair, which in cell-free DNA can vary by fragment length (as a consequence
of fragment origin)."""

import jtmethtools as jtm
from jtmethtools.util import plt_labels
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
from pathlib import Path

from dataclasses import field
import datargs


def nanmean_box_filter(a, n, mode="reflect"):
    """note, odd numbered `n` works weird, so odds will be rounded up to nearest even."""
    n += n % 2
    a = np.asarray(a, float)
    valid = np.isfinite(a)

    num = uniform_filter(np.where(valid, a, 0.0), size=n, mode=mode, )
    den = uniform_filter(valid.astype(float), size=n, mode=mode, )

    out = num / den
    out[den == 0] = np.nan  # all-NaN windows -> NaN
    return out


def get_counts(bamfn, width=300):
    """Get counts of read position (relative to fragment, not by reference) vs fragment length."""
    counts_r1_r2 = {k: np.zeros((width, width), dtype=np.uint32)
                    for k in ('met1', 'tot1', 'met2', 'tot2')}

    for aln in jtm.iter_bam(bamfn, paired_end=True):

        if aln.mapping_quality() < 20:
            continue
        if aln.a2 is None:
            continue

        metsymb = set(aln.metstr)

        if ('H' in metsymb) or ('U' in metsymb) or ('X' in metsymb):
            continue

        frag_len = (aln.reference_end - aln.reference_start)

        if frag_len >= width:
            continue

        for a in (aln.a, aln.a2):
            ri = ['2', '1'][a.is_read1]
            metstr = jtm.get_bismark_met_str(a)
            for q, r in a.get_aligned_pairs():

                if (q is None) or (r is None):
                    continue
                if metstr[q] in ('z', 'Z'):
                    frag_pos = q if a.is_forward else a.query_alignment_length - q
                    counts_r1_r2[f"met{ri}"][frag_pos, frag_len] += metstr[q] == 'Z'
                    counts_r1_r2[f"tot{ri}"][frag_pos, frag_len] += 1
    for i in ('1', '2'):
        counts_r1_r2[f'mean{i}'] = mt = counts_r1_r2[f'met{i}'] / counts_r1_r2[f'tot{i}']


    return counts_r1_r2


def plot_counts(counts_r1_r2, smooth=6, min_obs=20):

    figs = {}

    for i in (1, 2):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8),
                                       height_ratios=[0.6, .4])
        mt = nanmean_box_filter(counts_r1_r2[f'mean{i}'], smooth)
        mt[counts_r1_r2[f"tot{i}"] < min_obs] = np.nan
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
            np.nanmean(counts_r1_r2[f'mean{i}'], axis=0)
        )
        plt_labels('Fragment length', 'Prop. methylation', f"R{i} methylation by fragment length")
        plt.tight_layout()
        figs[i] = fig

    return figs


def main(bamfn, plot_out_dir, array_out_dir=None, sample_name=None, width=300, smooth=6, write_arrays=False):
    """Run counts + plotting workflow and write outputs.

    Files are written with prefix: outdir / (sample_name + '.').
    """
    bamfn = Path(bamfn)
    plot_out_dir = Path(plot_out_dir)


    plot_out_dir.mkdir(parents=True, exist_ok=True)


    sample = sample_name or bamfn.stem

    counts_r1_r2 = get_counts(bamfn, width=width)

    figs = plot_counts(counts_r1_r2, smooth=smooth)
    plot_prefix = plot_out_dir / f"{sample}."
    for i, fig in figs.items():
        fig.savefig(str(plot_prefix) + f"position-vs-frag-len.read{i}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    if write_arrays:
        array_out_dir = Path(array_out_dir)
        array_out_dir.mkdir(parents=True, exist_ok=True)
        arr_prefix = array_out_dir / f"{sample}."
        for k in ('met1', 'tot1', 'met2', 'tot2'):
            np.savetxt(str(arr_prefix) + f"read-pos-vs-frag-len.{k}.csv", counts_r1_r2[k], delimiter=',', fmt='%d')

    return counts_r1_r2


@datargs.argsclass(
    description="""\
Plot fragment position × fragment length methylation for read 1 & read 2.

Shows the effect of end repair, which in cell-free DNA can vary by fragment
length (as a consequence of fragment origin). Produces one PNG per read
(read1 and read2) in the plot output directory, and optionally writes the
underlying count matrices as CSVs.
"""
)
class CLIArgs:
    bamfn: Path = field(metadata=dict(
        required=True,
        help="Bismark BAM file (paired-end, with methylation call string).",
        aliases=['-b'],
    ))
    plot_out_dir: Path = field(metadata=dict(
        required=True,
        help="Directory for output PNG plots. Created if it does not exist.",
        aliases=['-p'],
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
        help="Box-filter window size for smoothing the methylation heatmap. "
             "Odd values are rounded up to the next even number.",
    ), default=6)



if __name__ == "__main__":
    args = datargs.parse(CLIArgs)

    main(
        bamfn=args.bamfn,
        plot_out_dir=args.plot_out_dir,
        array_out_dir=args.array_out_dir,
        sample_name=args.sample_name,
        width=args.width,
        smooth=args.smooth,
        write_arrays=args.array_out_dir is not None
    )