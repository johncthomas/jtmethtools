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
    need_CpH: bool = field( #todo test
        default=False,
        metadata=dict(
            required=False,
            help="If set, alignments with no CpH sites at all will be removed.",
            aliases=['--need-CH', ]
        ),
    )
    no_log_file: bool = field(
        default=False,
        metadata=dict(
            required=False,
            help="Disable writing log files to {outdir}/log/.",
        )
    )

def ttest_filter_bam():
    import os
    samstr = """@HD	VN:1.0	SO:none
@SQ	SN:1	LN:248956422
mapq20Unorder	18	1	100	20	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
unmethRev	18	1	100	42	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
methRev	18	1	101	32	10M	*	0	10	CCCCCCCCCC	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:HHHHHHHHHH
unmethFor	2	1	102	27	10M	*	0	10	TTTTTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:zzzzzzzzzz
methFor	2	1	103	22	10M	*	0	10	CCCCCCCCCC	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:ZZZZZZZZZZ
allA	2	1	104	17	10M	*	0	10	AAAAAAAAAA	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allC	2	1	105	12	10M	*	0	10	CCCCCCCCCC	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allG	2	1	106	12	10M	*	0	10	GGGGGGGGGG	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allT	2	1	107	12	10M	*	0	10	TTTTTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
oneCpG	2	1	107	12	10M	*	0	10	TTCGTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..z.......
twoCpG	2	1	107	12	10M	*	0	10	TTCGCGTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..z.z.....
mapq19	18	1	100	19	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
mapq20	18	1	100	20	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
oneCHH\t2\t1\t104\t22\t10M\t*\t0\t10\tTTCTTTTTTT\tABCDEFGHIJ\tNM:i:tag0\tMD:Z:tag1\tXM:Z:..H.......
twoCHH\t2\t1\t104\t22\t10M\t*\t0\t10\tTTCTTCTTTT\tABCDEFGHIJ\tNM:i:tag0\tMD:Z:tag1\tXM:Z:..H..H....
twoCHH1z\t2\t1\t104\t22\t10M\t*\t0\t10\tTTCTTCTTTC\tABCDEFGHIJ\tNM:i:tag0\tMD:Z:tag1\tXM:Z:..H..H...z
"""


    tmp_dir = Path.home() / 'tmp/test_filter_mCH_oqifjw'

    # delete tmp_dir to avoid stale data
    if tmp_dir.exists():
        for f in (tmp_dir/'log').glob('*'):
            f.unlink()
        (tmp_dir/'log').rmdir()
        for f in tmp_dir.glob("*"):
            f.unlink()
        tmp_dir.rmdir()

    tmp_dir.mkdir(parents=True, exist_ok=True)

    samfn = tmp_dir/ 'test_mdat_sam_vcoqhiwc.sam'

    with open(samfn, 'w') as f:
        f.write(samstr)


    print(str(samfn))
    args1 = parse(
        FilterCHArgs,
        [
            "-o", str(tmp_dir),
            "-b", str(samfn),
            "--se",
            # "--need-CH",
        ]
    )
    print('slkdihf', args1)
    main(args1)

    has_mCH = {'methRev', 'oneCHH', 'twoCHH', 'twoCHH1z'}
    no_mCH = {'unmethRev', 'unmethFor', 'allA', 'allC', 'allG', 'allT', 'oneCpG', 'twoCpG'}
    no_CH = {'allA', 'allC', 'allG', 'allT',  'oneCpG', 'twoCpG'}

    found = set()
    print(os.listdir(tmp_dir))
    with pysam.AlignmentFile(str(tmp_dir/"test_mdat_sam_vcoqhiwc.noCH.bam"), ) as b:
        for aln in b:
            if aln.query_name in has_mCH:
                raise AssertionError(f"Alignment {aln.query_name} should have been removed for having CH methylation.")
            found.add(aln.query_name)
        assert no_mCH.issubset(found), f"{found=}, {no_mCH=}"

    # test with --need-CH
    args2 = parse(
        FilterCHArgs,
        [
            "-o", str(tmp_dir),
            "-b", str(samfn),
            "--se",
            "--need-CH",
        ]
    )
    main(args2)
    found = set()
    with pysam.AlignmentFile(str(tmp_dir/"test_mdat_sam_vcoqhiwc.noCH.bam"), ) as b:
        for aln in b:
            if aln.query_name in has_mCH:
                raise AssertionError(f"Alignment {aln.query_name} should have been removed for having CH methylation.")
            if aln.query_name in no_CH:
                raise AssertionError(f"Alignment {aln.query_name} should have been removed for having no CH sites.")
            found.add(aln.query_name)
        exp = {'mapq19', 'mapq20', 'mapq20Unorder', 'unmethRev'}
        assert exp == found, f"{found=}, {exp=}"


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


    outfn = outdir / (str(bam.name)[:-4]+'.noCH.bam')

    remove_ch_methylation(bam, outfn, verbose=(not args.quiet), paired_end=args.pe, req_CpH=args.need_CpH)

    if log_id is not None:
        logger.remove(log_id)



if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print('Running tests...')
        ttest_filter_bam()
    else:
        main()