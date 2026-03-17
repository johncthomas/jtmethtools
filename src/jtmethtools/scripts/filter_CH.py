"""From a BAM file, write a new bam that does not contain alignments with CH methylation."""

import sys
from datetime import datetime
import pysam
from jtmethtools.alignments import get_bismark_met_str, iter_bam_segments
from datargs import argsclass, arg, parse
from dataclasses import field
from pathlib import Path
from loguru import logger


def remove_ch_methylation(bam_file: Path | str, output_file: Path | str,
                          paired_end: bool, versbose=True):
    """Write a new BAM file that does not contain alignments with CH methylation.
    A mCH in either segment of a paired-end read will cause the entire read to be discarded.
    """

    ch_meth_symbols = {'X', 'H', 'U'}

    with pysam.AlignmentFile(bam_file) as b:
        header_d = b.header.to_dict()
        header_d['PG'].append(
            {'ID': 'jtmethtools CH removal',
             'CL': f'remove_ch_methylation(bam_file={str(bam_file)}, output_file={str(output_file)}, paired_end={paired_end})'}
        )
        header = pysam.AlignmentHeader.from_dict(header_d)

    with pysam.AlignmentFile(output_file, "wb", header=header) as bam_out:
        total_alignments = 0
        written = 0
        ch_count = 0
        if versbose:
            logger.info(f"Writing to {output_file}")

        for alns in iter_bam_segments(bam_file, paired_end=paired_end):
            total_alignments += 1
            has_ch = False
            for alignment in alns:
                if alignment is None:
                    continue

                met_str = get_bismark_met_str(alignment)

                # Find symbols in met_str that overlap with bad symbols
                #   This method chosen after some benchmarking.
                if not set(met_str).isdisjoint(ch_meth_symbols):
                    has_ch = True
            if has_ch:
                ch_count += 1
            if not has_ch:
                written += 1
                for alignment in alns:
                    if alignment is None:
                        continue
                    bam_out.write(alignment)

        if versbose:
            logger.info(
                f"{ch_count}/{total_alignments} ({round(ch_count / total_alignments * 100, 1):}%) "
                f"{'paired ' if paired_end else ''}alignments with methylated CH discarded."
            )


@argsclass(
    description="""\
Write a new BAM file that does not contain alignments with CH methylation.
"""
)
class FilterCHArgs:
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
    pe: bool = field(
        default=False,
        metadata=dict(
            required=False,
            help="Whether the input BAM files are paired-end. Either --pe or --se must be set."
        )
    )
    se: bool = field(
        default=False,
        metadata=dict(
            required=False,
            help="Whether the input BAM files are single-end. Either --pe or --se must be set."
        )
    )
    no_log_file: bool = field(
        default=False,
        metadata=dict(
            required=False,
            help="Disable writing log files to {outdir}/log/.",
        )
    )


def main(args: FilterCHArgs = None):
    """Call remove_ch_methylation on each BAM file in the input, using given command-line args."""

    if args is None:
        args = parse(FilterCHArgs)

    if args.pe == args.se:
        logger.error("Exactly one of --pe or --se must be set.")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Configure logger: remove default sink, add colorized stderr sink
    logger.remove()
    if not args.quiet:
        logger.add(
            sys.stderr,
            level='INFO',
            colorize=True,
        )

    now = datetime.now().strftime("%Y%m%d_%H-%M-%S")

    for bam in args.bams:
        bam = Path(bam)
        log_id = None
        if not args.no_log_file:
            log_dir = outdir / "log"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{bam.name}.{now}.log"
            log_id = logger.add(log_path,
                                level='INFO',
                                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

        outfn = outdir / bam.name.replace('.bam', '.noCH.bam')
        remove_ch_methylation(bam, outfn, versbose=(not args.quiet), paired_end=args.pe)

        if log_id is not None:
            logger.remove(log_id)

if __name__ == '__main__':
    main()