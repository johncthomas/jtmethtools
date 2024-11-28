from pathlib import Path

import jtmethtools.images
import matplotlib.pyplot as plt
import numpy as np
import os

from pyarrow import compute

from jtmethtools.util import (
    read_array,
    write_array,
    logger,
    set_logger
)

import tempfile

from jtmethtools import alignment_data
logger.remove()
#set_logger('TRACE',)

TESTDIR = Path(os.path.dirname(__file__))

samstr = """@HD	VN:1.0	SO:coordinate
@SQ	SN:1	LN:248956422
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

regionsstr = "1\t90\t110\ttest1\n"
test_bam_fn = TESTDIR/'img-test.sam'
with open(test_bam_fn, 'w') as f:
    f.write(samstr)

    test_regions_fn = TESTDIR/'regions.bed'
    with open(test_regions_fn, 'w') as f:
        f.write(regionsstr)

read_data = alignment_data.process_bam(
    test_bam_fn,
    test_regions_fn,
    include_read_name=True
)


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
    img = jtmethtools.images.ImageMaker(window, start, end)

    fig, axes = img._plot_test_images()
    plt.savefig(TESTDIR/'channels.png', dpi=150, )


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








