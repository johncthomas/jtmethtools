
from pathlib import Path
import sys, os
import datargs
from dataclasses import dataclass, field
from datetime import datetime
import argparse
import jtmethtools


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S%f")

@datargs.argsclass(
    parser_params=dict(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ),
    description="""\
=======================================================================
Write methylation state of every CpG, every methylated CpH (by default) 
  and optionally unmethylated CpH.

The output is a pair of parquet files, and some metadata:
    
    locus_data.parquet: Methylation states, loci, PHRED, and other 
      information.
    
    read_data.parquet: Read information, such as read name, length, 
      mapping quality, etc. AlignmentIndex columns link the two tables.
      
    metadata.json: e.g. BAM file, regions file, date, etc.
=======================================================================
"""
)
class ArgsMethylationData:
    bam: Path = field(metadata=dict(
        help='BAM file will be converted to tables.',
        aliases=['-b'],
        required=True,
    ))
    outdir: Path =  field(metadata=dict(
        help='Directory to which the output dir will be written - so actual outdir is $outdir/$sample, '
             'where $sample comes from the input bam filename, or --sample_name.',
        aliases=['-o'],
        required=True,
    ))
    sample_name: str = field(default=None, metadata=dict(
        help='Sample name that will be used in output directory name. Default uses stem of input bamfn.',
        aliases=['-s']
    ))
    regions: Path = field(default=None, metadata=dict(
        help='Alignments that overlap with regions will be written to the table. '
             'If not provided (default), all alignments will be written.',
        aliases=['-r']
    ))
    # either --se or --pe for sing/paired end
    se: bool = field(default=False, metadata=dict(
        help='Set if the BAM file contains single-end reads. Either --se or --pe must be set.',
        aliases=['--single-end']
    ))
    pe: bool = field(default=False, metadata=dict(
        help='See --se.',
        aliases=['--paired-end']
    ))
    all_chrm: bool = field(default=False, metadata=dict(
        help='By default, only include reads from cannonical chromosomes. '
             'Set this to include all chromosomes (i.e. all unplaced scaffold and alts).',
        aliases=['-a']
    ))
    unmethylated_ch: bool = field(default=False, metadata=dict(
        help='Set to include unmethylated CH in output. By default, '
             'only methylated CH are included.',
        aliases = ['-c', '--ch']
    ))
    drop_mch_reads: bool = field(default=False, metadata=dict(
        help='Set to drop reads with any methylated CpH. By default, these are included.',
        aliases=['-d']
    ))
    min_mapq: int = field(default=20, metadata=dict(
        help='Minimum mapping quality to include read. Default is 20.',
        aliases=['-m']
    ))
    quiet: bool = field(default=False, metadata=dict(
        help='Set to silence info messages printed to STDOUT. Log file created either way.',
    ))



def bam_to_parquet(args:ArgsMethylationData):

    from jtmethtools.methylation_data import (
        process_bam_methylation_data,
        MethylationDataset,
        logger,
    )

    # check pe/se
    if args.se == args.pe:
        raise ValueError("Specify single-ended or paired-eneded with --se or --pe.")

    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Deal with output directory and sample name
    sample_name = args.sample_name if args.sample_name is not None else args.bam.stem
    outdir = args.outdir/sample_name
    outdir.mkdir(parents=True, exist_ok=True)

    # make log dir
    logdir = outdir/'log'
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)
    logger.add(logdir/f"{dt}.log", level='INFO')
    logger.add(sys.stderr, level='WARNING', colorize=True)
    if not args.quiet:
        logger.add(sys.stdout, level='INFO', colorize=True)
    logger.info('Write methylation dataset')
    logger.info(str(args))

    if args.regions:
        regions = jtmethtools.Regions.from_file(args.regions)
    else:
        regions = None

    data = process_bam_methylation_data(
        bamfn=args.bam,
        regions=regions,
        paired_end=args.pe,
        cannonical_chrm_only=not args.all_chrm,
        include_unmethylated_ch=args.unmethylated_ch,
        chunk_size=int(1e6),
        min_mapq=args.min_mapq,
        drop_methylated_ch_reads=args.drop_mCpH_reads
    )

    MethylationDataset(
        data.locus_data,
        data.read_data,
        data.processes
    ).write_to_dir(outdir)


def main():
    a = datargs.parse(ArgsMethylationData)
    bam_to_parquet(a)


if __name__ == '__main__':
    main()