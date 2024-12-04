import os

from pathlib import Path
from typing import Self, Collection, Tuple, Iterator

import numpy as np
import pyarrow as pa
from attrs import define
from jtmethtools.alignment_data import (
    TruePA, COLS_READ, PositionArray, Pathesque, AlignmentsData, \
    PAD_READID, NT_CODES, BISMARK_CODES, sampledown_rows, process_bam)
from jtmethtools.alignments import LociRange, Regions
from jtmethtools.util import write_array, read_array
from loguru import logger
from numpy.typing import NDArray
from pyarrow import compute as compute

PIXEL_DTYPE = np.float32

PixelArray = NDArray[PIXEL_DTYPE]

def plot_layer(img:PixelArray, ax=None, **imshow_kw):
    import matplotlib.pyplot as plt
    if ax:
        plt.sca(ax)
    kwargs = {'interpolation':'nearest', 'cmap':'gray'} | imshow_kw
    plt.imshow(img, **kwargs)


class ImageMaker:
    """Methods for generating image layers for different read stats,
    across given window.

    All methods for image layers start with "layer_*". Other useful
    methods will probably start with "get_*" or "plot_*".
    """
    # __slots__ = ['window', 'positions', 'unique_positions',
    #              'unique_rowids', 'position_indices', 'row_indices',
    #              '_readid_image', 'width', 'rows',
    #              'min_cpg', 'min_mapq']

    null_grey = 0.2 # value where a base exists but is negative
    strand_colours = (0.6, 1)

    def __init__(
            self,
            window:AlignmentsData,
            start:int,
            end:int,
            rows:int=100,
            seed=123987,
    ):

        self.window:AlignmentsData = window


        # add padding, and ensure gaps are shown by creating     a
        #  blank read that hits every position between the start
        #  and end of the window.
        self.width = width = end-start


        # If there's too many reads, filter to the required number
        read_ids = compute.unique(window.locus_data.readID).to_numpy()

        if read_ids.shape[0] > rows:
            np.random.seed(seed)
            read_ids = np.random.choice(read_ids, rows, replace=False)

            window.locus_data = window.locus_data.filter(
                compute.is_in(
                    window.locus_data.readID,
                    pa.Array.from_pandas(read_ids)
                )
            )

        pos = window.locus_data.position.to_numpy()

        # add the padding info
        rids = window.locus_data.readID.to_numpy()
        logger.trace(pos)
        logger.trace(rids)

        rows += 1 # the ones we expect plus the padder
        self.n_alignments = len(np.unique(rids))
        rids = np.concatenate([
            rids,
            np.ones(shape=(width,), dtype=rids.dtype) * PAD_READID
        ])


        pos = np.concatenate([
            pos, np.arange(start, end, dtype=pos.dtype)
        ])

        self.rows = rows

        # These values used by ._protoimage()
        unique_positions, position_indices = np.unique(pos, return_inverse=True)
        unique_rowids, row_indices = np.unique(rids, return_inverse=True)

        self.unique_positions:NDArray = unique_positions
        self.unique_rowids:NDArray = unique_rowids
        self.position_indices:NDArray = position_indices
        self.row_indices:NDArray = row_indices

        # image with readID as values, base for annotating read
        #  level values. Copys provided by method
        self._readid_image = self._protoimage(window.locus_data.readID)


    def _protoimage(self, states:pa.Array) -> PixelArray:
        """Create image (array) where values are given by state,
        and missing positions are np.nan
        """
        states = states.to_numpy()

        states = np.concatenate([
            states,
            np.zeros(shape=(self.width,), dtype=PIXEL_DTYPE),
        ])

        # Initialize the output array with NaN values
        output_array = np.full(
            (self.rows, len(self.unique_positions)),
            np.nan
        )

        # Populate the output array directly using the indices
        output_array[self.row_indices, self.position_indices] = states

        return output_array.astype(PIXEL_DTYPE)

    def _get_read_value(self, rid:int, col:str):
        """Get the value from `col` column in read table for the
        given read ID. Returns 0 when read ID is PAD_READID"""
        if rid == PAD_READID:
            return 0

        return self.window.read_data.get_value(rid, col)

    @staticmethod
    def _finish_image(image, fillval=0) -> PixelArray:
        """Fill missing (np.nan) positions in image with 0,
        or fillval & remove bottom row (which is only there
         to pad the width)."""
        image[np.isnan(image)] = fillval
        return image[:-1]

    @property
    def _read_id_image_copy(self) -> PixelArray:
        """Array with values as readID. Copied. For creating
        read level images."""
        return np.copy(self._readid_image)


    def layer_methylated_cpg(self) -> PixelArray:
        """1 where CpG is methylated, null_grey otherwise"""
        cpg_met = compute.equal(self.window.locus_data.methylation, pa.scalar(1))
        image = self._protoimage(cpg_met)
        image[image==0.] = self.null_grey

        return self._finish_image(image)

    def layer_methylated_other(self) -> PixelArray:
        """1 where non-cpg methylated, null_grey otherwise"""
        other_met = compute.greater(self.window.locus_data.methylation, pa.scalar(1))
        image = self._protoimage(other_met)
        image[image == 0.] = self.null_grey

        return self._finish_image(image)


    def layer_methylated_any(self) -> PixelArray:
        """1 where non-cpg methylated, null_grey otherwise"""
        other_met = compute.greater(self.window.locus_data.methylation, pa.scalar(0))
        image = self._protoimage(other_met)
        image[image == 0.] = self.null_grey

        return self._finish_image(image)


    def layer_bases(self) -> PixelArray:
        """Shades of grey for each colour"""
        image = self._protoimage(
            self.window.locus_data.nucleotide
        )

        logger.trace(image)
        # gives values (0.4, 0.6, 0.8, 1.)
        image = image/5 + 0.2

        return self._finish_image(image)


    def layer_bases_met_as_fifth(self) -> PixelArray:
        """The normal 4 plus methylated C"""
        image = self._protoimage(self.window.locus_data.nucleotide)
        image = self._finish_image(image)
        metimg = self.layer_methylated_any()
        mask = metimg == 1
        image[mask] = 5

        image = image/6 + 1/6

        # image is already finished
        return image


    def layer_strand(self) -> PixelArray:
        """light grey for rev strand, white for for strand."""
        image = self._read_id_image_copy
        for rid in self.unique_rowids:
            isfor = self._get_read_value(rid, 'is_forward')
            c = self.strand_colours[isfor]
            image[image==rid] = c

        return self._finish_image(image)


    def layer_mapping_quality(self) -> PixelArray:
        """Each read a shade of grey proportional to mapping quality"""
        image = self._read_id_image_copy
        for rid in self.unique_rowids:
            mapq = self._get_read_value(rid, 'mapping_quality')

            mapq += self.null_grey
            mapq /= self.window.read_data.max_mapq + self.null_grey

            image[image==rid] = mapq

        return self._finish_image(image)


    def layer_phred(self) -> PixelArray:
        image = self._protoimage(self.window.locus_data.phred_scores)
        image /= self.window.max_phred

        return self._finish_image(image)


    def _fill_reads_one_colour(self, c:float) -> PixelArray:
        image = self._protoimage(c)

        return self._finish_image(image)


    def get_pixarray_dict(
            self,
            image_types:Collection[str]
    ) -> dict[str, PixelArray]:
        """Get pixel arrays by layer name (with or without layer_*)"""

        images = {}
        for meth in image_types:
            if not meth.startswith('layer_'):
                meth = 'layer_'+meth
            try:
                img = getattr(self, meth)()
            except AttributeError:
                raise AttributeError(f"Unknown image type: {meth}")
            # key without "layer_"
            images[meth[6:]] = img
        return images


    # def plot_images(self, image_types:Collection[str]):
    #     import matplotlib.pyplot as plt
    #
    #     layers: dict[str, NDArray] = self.get_images(
    #         image_types
    #     )
    #
    #     n = len(layers)
    #     fig, axes = plt.subplots(1, n, figsize=(8*n, 1.6*3) )
    #     if isinstance(axes, plt.Axes):
    #         axes = [axes]
    #     logger.debug(axes)
    #     for axi, (k, img) in enumerate(layers.items()):
    #         plot_layer(img, ax=axes[axi])
    #         axes[axi].set_title(k)
    #     return fig, axes
    #
    # def _plot_test_images(self):
    #     import matplotlib.pyplot as plt
    #     layers:dict[str, NDArray] = self.get_images(
    #         ('bases', 'methylated_cpg', 'methylated_other', 'methylated_any',
    #          'bases_met_as_fifth', 'strand', 'mapping_quality', 'phred',
    #          )
    #     )
    #     n = len(layers)
    #     fig, axes = plt.subplots(1, n, figsize=(9*n, 1.8*3) )
    #     logger.debug(axes)
    #     for axi, (i, img) in enumerate(layers.items()):
    #
    #         plot_layer(img, ax=axes[axi])
    #         axes[axi].set_title(i)
    #     return fig, axes

    @staticmethod
    def available_layers() -> list[str]:
        self = ImageMaker
        return [l.replace('layer_', '') for l in dir(self) if l.startswith('layer_')]


@define
class Image:
    array:NDArray[PIXEL_DTYPE]
    channel_names: list[str]
    name:str='noname'

    def to_file(self, outfn:Pathesque,):
        """Write array with metadata.

        File will be gzipped if outfn endswith ".gz"."""

        write_array(self.array, outfn,
                    additional_metadata={
                        'channel_names':self.channel_names,
                        'name':self.name
                    })

    @classmethod
    def from_file(cls, fn:Pathesque):
        array, metadata = read_array(fn)
        return cls(array,
                   channel_names=metadata['channel_names'],
                   name=metadata['name']
                   )

    @classmethod
    def from_dict(cls, d:dict[str,NDArray], name='noname'):
        return cls(
            np.array(list(d.values())),
            channel_names=list(d.keys()),
            name=name
        )

    def to_dict(self) -> dict[str, PixelArray]:
        return {k:self.array[i] for i, k in enumerate(self.channel_names)}

    def plot(self, ):
        import matplotlib.pyplot as plt

        layers = self.to_dict()

        n = len(layers)
        fig, axes = plt.subplots(1, n, figsize=(8*n, 1.6*3) )
        if isinstance(axes, plt.Axes):
            axes = [axes]
        logger.debug(axes)
        for axi, (k, img) in enumerate(layers.items()):
            plot_layer(img, ax=axes[axi])
            axes[axi].set_title(k)
        return fig, axes

def generate_images_in_regions(
        data:AlignmentsData,
        regions:Regions,
        layers:Collection[str],
        min_cpg=0,
        min_mapq=0,
        max_other_met:int=np.inf,
        min_alignments=0,
        rows=500,
) -> Tuple[LociRange, dict[str, PixelArray], dict]:

    """Genarate a dictionary of image arrays."""

    for posrange in regions.iter():
        metadata = {}
        window = data.window(posrange)
        if min_cpg:
            window = window.filter_by_ncpg(min_cpg)
        if min_mapq:
            window = window.filter_by_mapping_quality(min_mapq)
        if max_other_met is not np.inf:
            window = window.filter_by_noncpg_met(max_other_met)

        imgfactory = ImageMaker(window, posrange.start, posrange.end, rows=rows)
        n_alignments = imgfactory.n_alignments
        if n_alignments <= min_alignments:
            logger.debug(f"Skipping {posrange.name} for too few alignments")
            continue

        imagedict = imgfactory.get_pixarray_dict(layers)
        images = list(imagedict.values())

        metadata['n_rows_with_alignment'] = n_alignments
        yield posrange, imagedict, metadata


def ttests(test_bam_fn, test_regions_fn, testoutdir,
           *, delete_first=False):
    import datetime
    testoutdir = Path(testoutdir)
    if delete_first:
        if os.path.isdir(testoutdir):
            for f in os.listdir(testoutdir):
                os.remove(testoutdir/f)
            os.rmdir(testoutdir, )
    # get current time for timedelta
    start = datetime.datetime.now()
    rd = process_bam(
        test_bam_fn,
        test_regions_fn
    )
    # time checkpoint
    next1 = datetime.datetime.now()
    print('Time to process:', next1 - start)

    rd.to_dir(testoutdir)

    next2 = datetime.datetime.now()
    print('Time to write:', next2 - next1)

    rd2 = AlignmentsData.from_dir(testoutdir)
    next3 = datetime.datetime.now()
    print('Time to read:' , next3 - next2)
    print('\n')
    rd.print_heads()
    rd2.print_heads()
    next4 = datetime.datetime.now()
    from collections import Counter
    image_shapes = Counter()
    for l, a, mdat in generate_images_in_regions(
        rd,
        regions=Regions.from_file(test_regions_fn),
        layers=('bases', 'bases_met_as_fifth', 'phred', 'mapping_quality'),
        rows=50
    ):
        a: dict[str, PixelArray]
        l: LociRange


        if mdat['n_rows_with_alignment'] > 20:

            print(l.name)
            for k, img in a.items():
                fn = testoutdir / f'image.{l.name}.{k}.tar.gz'
                write_array(img, fn, gzip=True)
            # a2 = read_array(fn, gzip=True)
            # print(a)
            # print('\n*****************\n\n')
            # print(a2)
            # plot_layer(a['phred'])
            import matplotlib.pyplot as plt
            plt.savefig(testoutdir/f'image.{l.name}.phred.png')
    next5 = datetime.datetime.now()
    print('Time to generate all images: ', next5 - next4)

    # img = ImageMaker(window)
    # #print(img.methylated_cpg())
    # print(img.methylated_cpg().shape)
    # import matplotlib.pyplot as plt
    # fig, axes = img._plot_test_images()
    # #imgd = img.get_images([])
    #
    # plt.savefig('/home/jcthomas/DevLab/NIMBUS/Figures/test_images.1.png')

if __name__ == '__main__':
    logger.remove()
    print(__file__)

    # get computer name
    comp_name = os.uname()[1]
    if comp_name == 'POS':
        home = Path('/home/jaytee')
    else:
        home = Path('/home/jcthomas/')
    #bm = home/'data/canary/sorted_qname/CMDL19003173_1_val_1_bismark_bt2_pe.deduplicated.bam'
    bm = home/'DevLab/NIMBUS/Data/test/bismark_10k.bam'
    rg = home/'DevLab/NIMBUS/Reference/regions-table.canary.4k.tsv'
    out = home/'DevLab/NIMBUS/Data/test/readdata_structure_test'
    ttests(bm, rg, out, delete_first=True)

