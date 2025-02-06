from pathlib import Path
import sys

def bam_to_parquet():

    args =  sys.argv[1:]

    if (len(args) != 3) or (args[0] in ('-h', '--help')):
        print('Write a BAM to a Parquet table.\n\n  Usage: jtm-write-loci-table BAM OUTDIR REGIONS-TABLE')
        exit(0)
    bam, outdir, regions =  args
    from jtmethtools.alignment_data import AlignmentsData, logger
    logger.info(str(args))
    data = AlignmentsData.from_bam(bam, regions)
    data.to_dir(outdir)

if __name__ == '__main__':
    bam_to_parquet()