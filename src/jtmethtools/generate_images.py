
from jtmethtools.images import *
import argparse
import datetime
import importlib.metadata
import jtmethtools

def run_image_gen__one_per_layer(
        bam:Pathesque,
        regions:Pathesque,
        outdir:Pathesque,
        layers:list[str],
        height:int,
        single_ended=False,
        **imggen_kwargs
):
    logger.info("run_image_gen__one_per_layer (that is one file per layer)")
    bam, regions, outdir = [Path(x) for x in (bam, regions, outdir)]

    logger.info(
        f"Args: {bam=}, {regions=}, {outdir=}, {layers=}, {height=}, {imggen_kwargs=}"
    )

    os.makedirs(outdir, exist_ok=True)

    start = datetime.datetime.now()
    rd = process_bam(
        bam,
        regions,
        single_ended=single_ended
    )
    next1 = datetime.datetime.now()
    logger.info('Time to process BAM:', start - next1)

    n_images = 0
    for loci, pixarrays, metadata in generate_images_in_regions(
        rd,
        regions=Regions.from_file(regions),
        layers=layers,
        rows=height,
        **imggen_kwargs
    ):
        pixarrays: dict[str, PixelArray]
        loci: LociRange

        for layer_name, img in pixarrays.items():
            n_images += 1
            fn = outdir / f'image.{loci.name}.{layer_name}.tar.gz'
            Image(img, (layer_name,)).to_file(fn, gzip=True)

    time_after_writing = datetime.datetime.now()
    logger.info(f"Time to write {n_images} images: {time_after_writing - next1}")



def print_available_layers():
    print('Available layer names:')
    l = sorted(ImageMaker.available_layers())
    sep = '\n - '
    print(' - ' + sep.join(l))

def check_layers(layers:list[str]):
    """Check specified """
    avail = ImageMaker.available_layers()
    bad_layer = [l for l in layers if l not in avail]
    if bad_layer:
        print(f"Layer name(s) {', '.join(bad_layer)} don't exist.\n")
        print_available_layers()
        return False
    return True


def plot_an_image(array_file:Pathesque, out_file:Pathesque):
    import matplotlib.pyplot as plt
    img = Image.from_file(array_file)
    img.plot()
    plt.savefig(out_file)


def main(args):

    if args.command == "run":
        del args.command

        if not check_layers(args.layers):
            exit(1)

        # set logger
        logger.remove()
        if not args.quiet:
            from jtmethtools.util import set_logger
            set_logger('INFO')
        del args.quiet

        # add a log file
        logfn = os.path.join(args.outdir, "00_log_{time}.txt")
        logger.add(logfn, level='INFO')

        logger.info(f"V{str(importlib.metadata.version(jtmethtools.__name__))}")

        run_image_gen__one_per_layer(**vars(args))

    elif args.command == "layers":
        print_available_layers()
    elif args.command == 'plot':
        plot_an_image(args.array_file, args.out_file)


def ttest():
    os.chdir('/home/jcthomas/DevLab/NIMBUS/Data/test')
    print('yes')
    run_image_gen__one_per_layer(
        bam=Path('bismark_10k.bam'),
        regions=Path('regions-table.4for10kbam.tsv'),
        outdir=Path('testt'),
        layers=['bases', 'phred', 'mapping_quality', 'methylated_cpg'],
        height=50,
        **{'min_cpg': 1, 'max_other_met': 1000, 'min_alignments': 0, 'min_mapq': 40},
    )
