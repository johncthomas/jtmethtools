from pathlib import Path

import jtmethtools.images
import matplotlib.pyplot as plt
import numpy as np
import os
from dataclasses import dataclass
from typing import Tuple
from pyarrow import compute

from jtmethtools.util import (
    read_array,
    write_array,
    logger,
    set_logger
)

import tempfile

from jtmethtools import alignment_data

from jtmethtools.alignments import *

logger.remove()
#set_logger('TRACE',)

TESTDIR = Path(os.path.dirname(__file__))

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
"""

fn_test_bam = TESTDIR / 'img-test.sam'
with open(fn_test_bam, 'w') as f:
    f.write(samstr)

fn_test_regions = TESTDIR/'regions.tsv'
with open(fn_test_regions, 'w') as f:
    f.write("Name	Chrm	Start	End	Threshold	Direction\n")
    f.write("test1\t1\t90\t110\t0\n")

read_data = alignment_data.process_bam(
    fn_test_bam,
    fn_test_regions,
    include_read_name=True,
    single_ended=True,
)

def test_iter_paired():
    samsorted = """\
@HD	VN:1.0	SO:queryname	SS:queryname:natural
@SQ	SN:1	LN:248956422
K00252:520:HCKCTBBXY:5:1101:1367:45396_1:N:0:ANCGATAG+GCCTCTAT	83	1	1113568	23	96M	=	1113307	-357	ACATATTAACAACAATAAACCTCAATTTTTCCCTCCTCAAAAAAATTTAATAAACGAATAAAAAACAAAAAAAACCTTAAAACAAAAACAACAATT	<A-AF77777--A7AFA--AJFAFF<AA<A<AFFFA-AJFFJJJJJJJJJJFFF7JJJFFA<JFJJJ-JJA-<JJJJFF<--JJJF<AFFJJFAJJ	NM:i:27	MD:Z:0G3G3G2G2G1G22G2G1G3G2G0G3G0G2G0G1G0G2G2G1G6G1G2G0G1G5G2	XM:Z:h...h...h..x..x.h......................x..h.h...h..hh..Zxh..hh.hh..x..h.h......h.h..xh.h.....x..	XR:Z:CT	XG:Z:GA
K00252:520:HCKCTBBXY:5:1101:1367:45396_1:N:0:ANCGATAG+GCCTCTAT	163	1	1113307	23	117M	=	1113568	357	TAAAACTCAAATACATTCTACCAATAATAAAACTCTAAAAACTCACCTAAACTCCACACTCCCCCTCGACACACGCAACCCTCAACCAACATAATAAAAAAAAATCACTCTTTCTTT	JF-F<FJJ7-AJ<FFJJJFAF<JAA-AFFFAFFF-A7<A--<FJFJFAAF<AAFFFFFJFF--7A<A<F<7JFFFJ--)AAFA)-AJAA--<<JAF<-7<-A-FA---7A777F7--	NM:i:26	MD:Z:1G7G9G3G5G1G4G2G0G7G1G11A5G15G3G0A0G1G0G1G1G2G1G0G1G0G10	XM:Z:.x.......x.........x...x.....h.h....x..hh.......x.h................Zx.....Z.........x...x.h.hh.h.h..h.hh..h..........	XR:Z:GA	XG:Z:GA
K00252:520:HCKCTBBXY:5:1101:1387:43884_1:N:0:ANCGATAG+GCCTCTAT	83	1	1034354	42	120M	=	1034332	-142	CCTCCTCCCACAATAAACCTAAAAAACTAAAACCAAAACCTAACACCCCCAAAACTATAACTAAATAAACTCTAAAAACACAACTACTATCCACCGCCCCAAAAATCCCAAAATAAAACG	JF7FFJJJJAFJAFAA7JJJJJJFJAJJFF-FJFAA<7AFJFAF7JJJFFF<JJJJJJA-FJJFFJ<F7JJJJ<JJAFFFFF7JJFJJ<JJJAJJJJJJJAJJJJJJJJJJJJJJJFAJJ	NM:i:42	MD:Z:11G2G1G3G0G1G0G3G0G1G2G0G0G4G0G8G0G0G2G1G3G0G0G1G6G1G0G0G4G2G2G3G8G0G0G0G5G0G2G0G0G0G2	XM:Z:...........z..h.h...xh.hh...xh.h..zxh....xh........xhh..x.h...xhh.h......x.hhh....x..x..x...z..Z.....xhhh.....xh..hhhh.Z	XR:Z:CT	XG:Z:GA
K00252:520:HCKCTBBXY:5:1101:1387:43884_1:N:0:ANCGATAG+GCCTCTAT	163	1	1034332	42	117M	=	1034354	142	TTCTACTCCAAAAAAATTCTACCCTCCTCCCACAATAAACCTAAAAAACTAAAACCAAAACCTAACACCCCCAAAACTATAACTAAATAAACTCTAAAAACACAACTACTATCCACC	JJJJJJJJJJFFFJJJJFJJJJJJJJFJJJJJJJAJJF-AJF7AJFJJAJF<<FFJJJFFFJFJFJFFJJJJJAFAFFAAFAJJJJ7AJJFA7FFJJJJJF<FF<A<<FAF<AJFFJ	NM:i:37	MD:Z:9G0G0G1G6G12G2G1G3G0G1G0G3G0G1G2G0G0G4G0G8G0G0G2G1G3G0G0G1G6G1G0G0G4G2G2G3G2	XM:Z:.........zxh.h......x............z..h.h...xh.hh...xh.h..zxh....xh........xhh..x.h...xhh.h......x.hhh....x..x..x...z..	XR:Z:GA	XG:Z:GA
K00252:520:HCKCTBBXY:5:1101:2361:48245_1:N:0:AGCGATAG+GCCTCTAT	99	1	1417392	40	96M	=	1417551	275	GTTTTTGTTGAAAATTTTGGAAGTTTGGGATTTTAATTATTTTTTGTTATTAATAAGATTTTGTTTAGTTTAAATTAATTTAGAATTGGTATAGAT	FA-<JJJAJAJJJJF-<-<JJJJ<AJFJJJFFJJFJFAJJ-FJ<FJJJJJJJJJFJJJJF-JAAFJFFAJFFJJJJFJJJJJJJ<AAAJ<JJJJFF	NM:i:11	MD:Z:1C12C0C17C6C0C4C11C0C9C8C17	XM:Z:.h............hh.................h......hh....h...........hh.........h........h.................	XR:Z:CT	XG:Z:CT
K00252:520:HCKCTBBXY:5:1101:2361:48245_1:N:0:AGCGATAG+GCCTCTAT	147	1	1417551	40	116M	=	1417392	-275	TAATGATCGTAGGGGGGAGGGGGTTTAAAATTTGTTTTAAGTGTTTATGTACGGAAATTGGTTTGGGTGTTTTGGTTTATACGTTATTTTGTGTTATATTTTTGAAATAAGGGATT	-7-<-A-7A7)<-)-<--F7--FAJJAA<AFJJJJFF<---<7--------7------A-<AAF7---JFFJ77F<F7-F--F<A-JFFF<F<A--F-FJJJJF<<A<AAA-A-JA	NM:i:26	MD:Z:3C0T1C17C5C0C2C1C3C3C0C11C3C0C6C0C4C3C6C0C6C3C0C7C6C0C0	XM:Z:...h..xZ................h.....hx..h.h.......hh.....Z.....x...hx......hh....h...h.Z....hh......h...hh.......h......hx	XR:Z:GA	XG:Z:CT
"""


    fn_sorted = TESTDIR / 'paired-test.sam'
    with open(fn_sorted, 'w') as f:
        f.write(samsorted)

    i = None
    for i, aln in enumerate(iter_bam(fn_sorted, paired_end=True)):
        pass
    assert i == 2
    for i, aln in enumerate(iter_bam(fn_sorted, paired_end=False)):
        pass
    assert i == 5


def test_iter_unsorted_paired():
    samunsorted = """\
@HD	VN:1.0	SO:none	SS:none:none
@SQ	SN:1	LN:248956422
K00252:520:HCKCTBBXY:5:1101:2361:48245_1:N:0:AGCGATAG+GCCTCTAT	99	1	1417392	40	96M	=	1417551	275	GTTTTTGTTGAAAATTTTGGAAGTTTGGGATTTTAATTATTTTTTGTTATTAATAAGATTTTGTTTAGTTTAAATTAATTTAGAATTGGTATAGAT	FA-<JJJAJAJJJJF-<-<JJJJ<AJFJJJFFJJFJFAJJ-FJ<FJJJJJJJJJFJJJJF-JAAFJFFAJFFJJJJFJJJJJJJ<AAAJ<JJJJFF	NM:i:11	MD:Z:1C12C0C17C6C0C4C11C0C9C8C17	XM:Z:.h............hh.................h......hh....h...........hh.........h........h.................	XR:Z:CT	XG:Z:CT
K00252:520:HCKCTBBXY:5:1101:1367:45396_1:N:0:ANCGATAG+GCCTCTAT	83	1	1113568	23	96M	=	1113307	-357	ACATATTAACAACAATAAACCTCAATTTTTCCCTCCTCAAAAAAATTTAATAAACGAATAAAAAACAAAAAAAACCTTAAAACAAAAACAACAATT	<A-AF77777--A7AFA--AJFAFF<AA<A<AFFFA-AJFFJJJJJJJJJJFFF7JJJFFA<JFJJJ-JJA-<JJJJFF<--JJJF<AFFJJFAJJ	NM:i:27	MD:Z:0G3G3G2G2G1G22G2G1G3G2G0G3G0G2G0G1G0G2G2G1G6G1G2G0G1G5G2	XM:Z:h...h...h..x..x.h......................x..h.h...h..hh..Zxh..hh.hh..x..h.h......h.h..xh.h.....x..	XR:Z:CT	XG:Z:GA
K00252:520:HCKCTBBXY:5:1101:1387:43884_1:N:0:ANCGATAG+GCCTCTAT	83	1	1034354	42	120M	=	1034332	-142	CCTCCTCCCACAATAAACCTAAAAAACTAAAACCAAAACCTAACACCCCCAAAACTATAACTAAATAAACTCTAAAAACACAACTACTATCCACCGCCCCAAAAATCCCAAAATAAAACG	JF7FFJJJJAFJAFAA7JJJJJJFJAJJFF-FJFAA<7AFJFAF7JJJFFF<JJJJJJA-FJJFFJ<F7JJJJ<JJAFFFFF7JJFJJ<JJJAJJJJJJJAJJJJJJJJJJJJJJJFAJJ	NM:i:42	MD:Z:11G2G1G3G0G1G0G3G0G1G2G0G0G4G0G8G0G0G2G1G3G0G0G1G6G1G0G0G4G2G2G3G8G0G0G0G5G0G2G0G0G0G2	XM:Z:...........z..h.h...xh.hh...xh.h..zxh....xh........xhh..x.h...xhh.h......x.hhh....x..x..x...z..Z.....xhhh.....xh..hhhh.Z	XR:Z:CT	XG:Z:GA
K00252:520:HCKCTBBXY:5:1101:1367:45396_1:N:0:ANCGATAG+GCCTCTAT	163	1	1113307	23	117M	=	1113568	357	TAAAACTCAAATACATTCTACCAATAATAAAACTCTAAAAACTCACCTAAACTCCACACTCCCCCTCGACACACGCAACCCTCAACCAACATAATAAAAAAAAATCACTCTTTCTTT	JF-F<FJJ7-AJ<FFJJJFAF<JAA-AFFFAFFF-A7<A--<FJFJFAAF<AAFFFFFJFF--7A<A<F<7JFFFJ--)AAFA)-AJAA--<<JAF<-7<-A-FA---7A777F7--	NM:i:26	MD:Z:1G7G9G3G5G1G4G2G0G7G1G11A5G15G3G0A0G1G0G1G1G2G1G0G1G0G10	XM:Z:.x.......x.........x...x.....h.h....x..hh.......x.h................Zx.....Z.........x...x.h.hh.h.h..h.hh..h..........	XR:Z:GA	XG:Z:GA
K00252:520:HCKCTBBXY:5:1101:1387:43884_1:N:0:ANCGATAG+GCCTCTAT	163	1	1034332	42	117M	=	1034354	142	TTCTACTCCAAAAAAATTCTACCCTCCTCCCACAATAAACCTAAAAAACTAAAACCAAAACCTAACACCCCCAAAACTATAACTAAATAAACTCTAAAAACACAACTACTATCCACC	JJJJJJJJJJFFFJJJJFJJJJJJJJFJJJJJJJAJJF-AJF7AJFJJAJF<<FFJJJFFFJFJFJFFJJJJJAFAFFAAFAJJJJ7AJJFA7FFJJJJJF<FF<A<<FAF<AJFFJ	NM:i:37	MD:Z:9G0G0G1G6G12G2G1G3G0G1G0G3G0G1G2G0G0G4G0G8G0G0G2G1G3G0G0G1G6G1G0G0G4G2G2G3G2	XM:Z:.........zxh.h......x............z..h.h...xh.hh...xh.h..zxh....xh........xhh..x.h...xhh.h......x.hhh....x..x..x...z..	XR:Z:GA	XG:Z:GA
K00252:520:HCKCTBBXY:5:1101:2361:48245_1:N:0:AGCGATAG+GCCTCTAT	147	1	1417551	40	116M	=	1417392	-275	TAATGATCGTAGGGGGGAGGGGGTTTAAAATTTGTTTTAAGTGTTTATGTACGGAAATTGGTTTGGGTGTTTTGGTTTATACGTTATTTTGTGTTATATTTTTGAAATAAGGGATT	-7-<-A-7A7)<-)-<--F7--FAJJAA<AFJJJJFF<---<7--------7------A-<AAF7---JFFJ77F<F7-F--F<A-JFFF<F<A--F-FJJJJF<<A<AAA-A-JA	NM:i:26	MD:Z:3C0T1C17C5C0C2C1C3C3C0C11C3C0C6C0C4C3C6C0C6C3C0C7C6C0C0	XM:Z:...h..xZ................h.....hx..h.h.......hh.....Z.....x...hx......hh....h...h.Z....hh......h...hh.......h......hx	XR:Z:GA	XG:Z:CT
"""
    fn_unsort = TESTDIR / 'paired-unsorted-test.sam'

    with open(fn_unsort, 'w') as f:
        f.write(samunsorted)
    for i, aln in enumerate(iter_bam(fn_unsort, paired_end=True)):
        assert aln.a2 is not None
        pass
    assert i == 2


def test_read_write_array():
    arr = np.array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    write_array(arr, TESTDIR / 'arrtest.tar',
                additional_metadata={'thing':1})
    arr_read, metad = read_array(TESTDIR / 'arrtest.tar')

    assert np.all(arr == arr_read)
    assert metad['thing'] == 1

    arr2 = arr.astype(np.float32)
    write_array(arr2, TESTDIR / 'arrtest2.tar')
    arr2_read, _ = read_array(TESTDIR/'arrtest2.tar')
    assert arr2_read.dtype == np.float32

    # check it works with temp files.
    with tempfile.NamedTemporaryFile('w') as tmpf:
        write_array(arr2, tmpf.name)
        arr2_read2, _ = read_array(tmpf.name)

        assert np.all(arr == arr2_read2)


def test_save_images():
    start, end = 90, 110
    window = read_data.window(start=start, end=end, chrm='1')
    img = jtmethtools.images.ImageMaker(window, start, end, rows=20)

    from jtmethtools.images import (
        generate_images_in_regions,

        plot_layer
    )

    layer_methods = img.available_layers()
    n = len(layer_methods)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    for i, (k, layer) in enumerate(
            img.get_pixarray_dict(layer_methods).items()
    ):
        ax = axes[i]
        plot_layer(layer, ax=ax)
        ax.set_xlabel(k)

    plt.savefig(TESTDIR / f'image_layers_test_bam.png')

    # fig, axes = img._plot_test_images()
    # plt.savefig(TESTDIR/'channels.png', dpi=150, )


def check_filtering(rd, rid_in, rid_not_in):
    ldf = rd.locus_data.table.to_pandas()
    rdf = rd.read_data.table.to_pandas()

    rids = ldf.readID.unique()
    rnames = set(rdf.set_index('readID').loc[rids, 'read_name'].values)

    assert rid_not_in not in rnames
    assert rid_in in rnames


def test_filter_read_data():

    filt_ncpg = read_data.filter_by_ncpg(2)
    check_filtering(filt_ncpg,'twoCpG', 'oneCpG' )

    filt_mapq = read_data.filter_by_mapping_quality(20)
    check_filtering(filt_mapq, 'mapq20', 'mapq19')

    filt_meth = read_data.filter_by_noncpg_met(max_noncpg=0)
    check_filtering(filt_meth, 'unmethRev', 'methRev')


def test_merge_paired_alignment_values():
    @dataclass
    class A:
        metstr: str
        query: str
        query_qualities: list[int]
        aligned_pairs: list[Tuple[int, int]]

        def get_tags(self):
            return [None, None, ('XM', self.metstr)]


    testa1 = A(
        metstr="ACDE",
        query='acde',
        query_qualities=[40, 40, 10, 40],
        aligned_pairs=[
            (0, 10),
            (None, 11),
            (1, 12),
            (2, 13),
            (3, None)
        ]
    )

    testa2 = A(
        metstr="FGHI",
        query='fghi',
        query_qualities=[10, 40, 40, 40],
        aligned_pairs=[
            (0, 12),
            (1, 13),
            (2, 14),
            (3, 15)
        ]
    )
    testaln = Alignment(
        a=testa1,
        a2=testa2
    )

    res = testaln.get_locus_values(use_quality_profile=True)
    assert  res == {'phreds': {10: 40, 14: 40, 15: 40, 12: 37, 13: 34},
                    'nucleotides': {10: 'a', 14: 'h', 15: 'i', 12: 'c', 13: 'g'},
                    'methylations': {10: 'A', 14: 'H', 15: 'I', 12: 'C', 13: 'G'}}






