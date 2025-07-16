
from jtmethtools.scripts.write_methylation_data import *
from jtmethtools.methylation_data import logger
import pandas as pd
script_path = Path(os.path.abspath(__file__)).parent



def test(quiet=True):
    #home = Path.home()
    samhead = """\
@HD	VN:1.0	SO:coordinate
@SQ	SN:1	LN:248956422
@SQ	SN:2	LN:248956422
@SQ	SN:other	LN:248956422
@SQ	SN:4	LN:248956422
"""
    sam_alignments = """\
FullLenn	2	1	5	12	22M	*	0	10	CGCGCGCGCGCGCGCGCGCGAA	ABCDEFGHIJABCDEFGHIJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:Z.Z.Z.Z.Z.Z.Z.Z.Z.Z...
FullChrm2	2	2	3	12	22M	*	0	10	AACGCGCGCGCGCGCGCGCGCG	ABCDEFGHIJABCDEFGHIJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..Z.Z.Z.Z.Z.Z.Z.Z.Z.Z.
NonCanon	2	other	3	12	22M	*	0	10	AACGCGCGCGCGCGCGCGCGCG	ABCDEFGHIJABCDEFGHIJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..Z.Z.Z.Z.Z.Z.Z.Z.Z.Z.
HasCH	2	4	3	12	22M	*	0	10	AATATATAGAGAAAAAAAAAAA	ABCDEFGHIJABCDEFGHIJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..h.h.h.H.H...........
"""

    tmp_dir = script_path/ 'tmp/'

    tmp_dir.mkdir(exist_ok=True, parents=True)
    samfn2 = tmp_dir/'tst2_full.sam'
    with open(samfn2, 'w') as f:
        f.write(samhead + sam_alignments)


    rgn_fn = tmp_dir / 'test_region.single.tsv'

    pd.DataFrame({
        'Chrm': ['1',],
        'Start': [0, ],
        'End': [248956422, ],
        'Name': ['test_region', ],
    }).to_csv(rgn_fn, sep='\t', index=False)

    outd = tmp_dir / 'test-methylation-data/'
    logger.info(outd)

    logger.remove()
    if not quiet:
        logger.add(sys.stdout, level='INFO', colorize=True)

    args1 = ArgsMethylationData(
        bam=samfn2,
        outdir=outd/'t1_defaults',
        regions=None,
        single_end=True,
        all_chrm=False,
        unmethylated_ch=False,
        quiet=True
    )

    bam_to_parquet(args1)

    logger.remove()
    if not quiet:
        logger.add(sys.stdout, level='INFO', colorize=True)

    # test all_chrm and unmethylated_ch
    args2 = ArgsMethylationData(
        bam=samfn2,
        outdir=outd/'t2_all_chrm_unmethylated',
        regions=None,
        single_end=True,
        all_chrm=True,
        unmethylated_ch=True,
        quiet=True
    )

    bam_to_parquet(args2)

    logger.remove()
    if not quiet:
        logger.add(sys.stdout, level='INFO', colorize=True)

    # test regions
    args3 = ArgsMethylationData(
        bam=samfn2,
        outdir=outd/'t3_regions',
        regions=rgn_fn,
        single_end=True,
        all_chrm=False,
        unmethylated_ch=False,
        quiet=True
    )
    bam_to_parquet(args3)

    logger.remove()
    if not quiet:
        logger.add(sys.stdout, level='INFO', colorize=True)

#ttest(quiet=False)

# # output of this was manually checked with
# tables = {}
# for paff in glob(str(d)+'/*/locus_data.parquet'):
#     t = pd.read_parquet(paff)
#     k = paff.split('/')[-2]
#     tables[k] = t
#
# for k, t in tables.items():
#     print(k)
#     display(t)
