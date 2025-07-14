
from pathlib import Path
import sys, os
import datargs
from dataclasses import dataclass, field
from datetime import datetime

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S%f")

@datargs.argsclass(
    description="""\
Take a BAM and write every nucleotide as a 
"""
)
class WADArgs:
    bam: Path = field(metadata=dict(
        help='BAM file will be converted to tables.',
        aliases=['-b']
    ))
    outdir: Path =  field(metadata=dict(
        help='Directory to which the tables will be written.',
        aliases=['-o']
    ))
    regions: Path = field(metadata=dict(
        default=None,
        help='Alignments that overlap with regions will be written to the table. '
             'If not provided (default), all alignments will be written.',
        aliases=['-r']
    ))

    # optionals
    quiet: bool = field(default=False, metadata=dict(
        help='Alignments that overlap with regions will be written to the table.',
    ))



def bam_to_parquet(args:WADArgs):

    from jtmethtools.alignment_data import AlignmentsData, logger
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(args.outdir/f"log.{dt}.log", level='INFO')
    if not args.quiet:
        logger.add(print, level='INFO')
    logger.info('Write alignment dataset')
    logger.info(str(args))

    data = AlignmentsData.from_bam(args.bam, args.regions)
    data.to_dir(args.outdir)


def main():
    a = datargs.parse(WADArgs)
    bam_to_parquet(a)


if __name__ == '__main__':
    main()