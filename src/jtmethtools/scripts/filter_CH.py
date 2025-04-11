"""From a BAM file, write a new bam that does not contain alignments with CH methylation."""

import pysam
from jtmethtools.alignments import get_bismark_met_str
from datargs import argsclass, arg, parse
from dataclasses import field
from pathlib import Path
from pysam import AlignmentFile


def remove_ch_methylation(bam_file: Path|str, output_file: Path|str, versbose=True):
    """Remove alignments with CH methylation from a BAM file."""
    ch_meth_symbols = {'X', 'H', 'U'}
    with pysam.AlignmentFile(bam_file, "rb") as bam_in, \
         pysam.AlignmentFile(output_file, "wb", template=bam_in) as bam_out:
        total_alignments = 0
        written = 0
        has_ch = 0
        if versbose:
            print(f"Writing {output_file}")
        for alignment in bam_in:
            total_alignments += 1
            # Get the methylation string
            met_str = get_bismark_met_str(alignment)
            # Write to out file if no overlapping symbols with CH methylation
            if set(met_str).isdisjoint(ch_meth_symbols):
                written += 1
                bam_out.write(alignment)
            else:
                has_ch += 1
        if versbose:
            print(f"{has_ch}/{total_alignments} ({round(has_ch/total_alignments*100, 1):}%) alignments with methylated CH discarded. ")


@argsclass(
    description="""\
Write a new BAM file that does not contain alignments with CH methylation.
"""
)
class ReadStatsArgs:
    outdir: Path = field(metadata=dict(
        required=True,
        help='Output directory. Files will be written with "noCH.bam" suffix.',
        aliases=['-o',],
    ))
    # positional arg at the end
    bams: list[Path] = arg(
        '-b',
        metavar='BAM',
        nargs='+',
        help="Bismark BAM files.",
    )
    quiet: bool = field(
        metadata=dict(
            required=False,
            help="Don't print logging messages."
        )
    )

def main():
    args = parse(ReadStatsArgs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for bam in args.bams:
        outfn = outdir / bam.name.replace('.bam', '.noCH.bam')
        remove_ch_methylation(bam, outfn, versbose=(not args.quiet))

main()