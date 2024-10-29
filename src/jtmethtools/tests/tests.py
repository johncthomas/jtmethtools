from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
TESTDIR = Path(os.path.dirname(__file__))
from jtmethtools.util import (
    read_array,
    write_array,
    logger,
    set_logger
)
from jtmethtools import alignment_data
logger.remove()
#set_logger('TRACE',)

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


def test_save_images():
    samstr = """@HD	VN:1.0	SO:coordinate
@SQ	SN:1	LN:248956422
unmethRev	18	1	100	42	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
methRev	18	1	101	32	10M	*	0	10	CCCCCCCCCC	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:HHHHHHHHHH
unmethFor	2	1	102	27	10M	*	0	10	TTTTTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:zzzzzzzzzz
methFor	2	1	103	22	10M	*	0	10	CCCCCCCCCC	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:ZZZZZZZZZZ
allA	2	1	104	17	10M	*	0	10	AAAAAAAAAA	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allC	2	1	105	12	10M	*	0	10	CCCCCCCCCC	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allG	2	1	106	12	10M	*	0	10	GGGGGGGGGG	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allT	2	1	107	12	10M	*	0	10	TTTTTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........\n"""

    regionsstr = "1\t90\t110\ttest1\n"
    test_bam_fn = TESTDIR/'img-test.sam'
    with open(test_bam_fn, 'w') as f:
        f.write(samstr)

    test_regions_fn = TESTDIR/'regions.bed'
    with open(test_regions_fn, 'w') as f:
        f.write(regionsstr)

    rd = alignment_data.process_bam(
        test_bam_fn,
        test_regions_fn
    )

    rd.print_heads(100)

    # rd.to_dir(TESTDIR/'data')
    #
    # rd2 = alignment_data.AlignmentsData.from_dir(TESTDIR)

    window = rd.window(90, 110, '1')
    img = alignment_data.ImageMaker(window, image_types=[])

    fig, axes = img._plot_test_images()
    plt.savefig(TESTDIR/'channels.png', dpi=150, )



