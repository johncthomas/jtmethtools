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
                          paired_end: bool, req_CpH:bool, verbose=True):
    """Write a new BAM file that does not contain alignments with CH methylation.
    A mCH in either segment of a paired-end read will cause the entire read to be discarded.


    """

    ch_meth_symbols = {'X', 'H', 'U'}
    ch_symbols = {'X', 'H', 'U', 'x', 'h', 'u'}

    with pysam.AlignmentFile(bam_file) as b:
        header_d = b.header.to_dict()
        if 'PG' not in header_d:
            header_d['PG'] = []
        header_d['PG'].append(
            {'ID': 'jtmethtools CH removal',
             'CL': f'remove_ch_methylation(bam_file={str(bam_file)}, output_file={str(output_file)}, paired_end={paired_end})'}
        )
        header = pysam.AlignmentHeader.from_dict(header_d)

    with pysam.AlignmentFile(output_file, "wb", header=header) as bam_out:
        total_alignments = 0
        written = 0
        mCH_count = 0
        noCH_count = 0
        if verbose:
            logger.info(f"Writing to {output_file}")

        for alns in iter_bam_segments(bam_file, paired_end=paired_end):
            total_alignments += 1
            has_mCH = False
            no_CH = False
            for alignment in alns:
                if alignment is None:
                    continue

                met_str = get_bismark_met_str(alignment)
                set_met = set(met_str)
                if met_str is None:
                    continue

                # Find symbols in met_str that overlap with bad symbols
                #   This method chosen after some benchmarking.
                if not set_met.isdisjoint(ch_meth_symbols):
                    has_mCH = True
                    break

                if req_CpH and set_met.isdisjoint(ch_symbols):
                    no_CH = True
                    break

            if has_mCH:
                mCH_count += 1
            elif req_CpH and no_CH:
                noCH_count += 1
            else:
                written += 1
                for alignment in alns:
                    if alignment is None:
                        continue
                    bam_out.write(alignment)

        if verbose:
            if total_alignments == 0:
                logger.info("No alignments found in input BAM.")
            else:
                logstr = (f"{written}/{total_alignments} ({round(written / total_alignments * 100, 1):}%) "
                    f"{'paired ' if paired_end else ''}alignments written. {mCH_count} alignments methylated CpH removed.")
                if req_CpH:
                    logstr += f" {noCH_count} alignments with no CpH context removed ."
                logger.info(logstr)


@argsclass(
    description="""\
Write a new BAM file that does not contain alignments with CH methylation.
In EM-seq data, CH methylation is considered to be a technical artifact, 
that indicates a failure of the whole read.

Optionally remove reads with no CpH sites at all, as these 
cannot be assessed for the presence of CH methylation.
"""
)
class FilterCHArgs:
    outdir: Path = field(metadata=dict(
        required=True,
        help='Output directory. Files will be written with "noCH.bam" suffix.',
        aliases=['-o',],
    ))
    # positional arg at the end
    bam: Path = arg(
        '-b',
        metavar='BAM',
        help="Bismark BAM file.",
    )
    quiet: bool = field(
        metadata=dict(
            required=False,
            help="Don't print logging messages."
        ), default=False
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
    req_CpH: bool = field( #todo test
        default=False,
        metadata=dict(
            required=False,
            help="If set, alignments with no CpH sites at all will be removed."
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
    """Call remove_ch_methylation on BAM file, using given command-line args."""

    if args is None:
        args = parse(FilterCHArgs)

    # Configure logger: remove default sink, add colorized stderr sink
    logger.remove()
    if not args.quiet:
        logger.add(
            sys.stdout,
            level='INFO',
            colorize=True,
        )
    logger.info(str(args))
    if args.pe == args.se:
        logger.error("Exactly one of --pe or --se must be set.")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y%m%d_%H-%M-%S")

    bam = Path(args.bam)
    log_id = None
    if not args.no_log_file:
        log_dir = outdir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{bam.name}.{now}.log"
        log_id = logger.add(log_path,
                            level='INFO')

    outfn = outdir / bam.name.replace('.bam', '.noCH.bam')
    remove_ch_methylation(bam, outfn, verbose=(not args.quiet), paired_end=args.pe)

    outfn = outdir / (str(bam.name)[:-4]+'.noCH.bam')


    if log_id is not None:
        logger.remove(log_id)



if __name__ == '__main__':
    main()