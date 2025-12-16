import jtmethtools as jtm
from pathlib import Path
import pysam
from argparse import ArgumentParser

def filter_bam_by_length(
        bamfn: str|Path,
        outfn: str|Path,
        min_len: int,
        max_len: int,
        paired: bool = True,
):
    infile = pysam.AlignmentFile(bamfn, 'rb')
    count = 0
    with pysam.AlignmentFile(outfn, 'wb', header=infile.header) as outf:
        for aln in jtm.iter_bam(bamfn, paired_end=paired):

            positions = aln.a.positions
            if paired and (aln.a2 is not None):
                positions += aln.a2.positions

            l = max(positions) - min(positions)

            if min_len <= l <= max_len:
                count += 1
                outf.write(aln.a)
                if aln.a2 is not None:
                    outf.write(aln.a2)

    print(f'Wrote {count} reads to {outfn}')

if __name__ == '__main__':
    parser = ArgumentParser(
        description="Extract reads within a specified length range from a BAM file."
    )
    parser.add_argument(
        '-i', '--input-bam', required=True, type=str, help='Input BAM file.'
    )
    parser.add_argument(
        '-o', '--output-bam', required=True, type=str, help='Output BAM file for extracted reads.'
    )
    parser.add_argument(
        '--min-len', type=int, default=0, help='Minimum read length to include. Default is 0.'
    )
    parser.add_argument(
        '--max-len', type=int, default=1e10, help='Maximum read length to include. Default is 1,000,000.'
    )
    parser.add_argument(
         '--pe', action='store_true', help='Set if the BAM file contains paired-end reads.'
    )
    parser.add_argument(
         '--se', action='store_true', help='Set if the BAM file contains single-end reads.'
    )

    args = parser.parse_args()

    if (not (args.pe or args.se)) or (args.pe and args.se):
        parser.error('Either --se or --pe for single/paired-ends must be specified.')

    filter_bam_by_length(
        bamfn=args.input_bam,
        outfn=args.output_bam,
        min_len=args.min_len,
        max_len=args.max_len,
        paired=args.pe,
    )
