import os

import numpy as np

from jtmethtools.images import *
import argparse
import datetime


def run_image_gen(
        bam:Pathesque,
        regions:Pathesque,
        outdir:Pathesque,
        layers:list[str],
        height:int,
        single_eneded=False,
        **imggen_kwargs
):

    bam, regions, outdir = [Path(x) for x in (bam, regions, outdir)]

    logger.info(
        f"{bam=}, {regions=}, {outdir=}, {layers=}"
    )

    os.makedirs(outdir, exist_ok=True)

    start = datetime.datetime.now()
    rd = process_bam(
        bam,
        regions,

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
            write_array(img, fn, gzip=True)

    time_after_writing = datetime.datetime.now()
    logger.info(f"Time to write {n_images} images: {time_after_writing - next1}")


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Generate images from BAM files oni specified regions, or view available layers."
    )

    subparsers = argparser.add_subparsers(
        dest="command", required=True,
    )

    # Subparser for 'run' command
    run_parser = subparsers.add_parser(
        "run",
        help="Run the image generation process."
    )
    run_parser.add_argument(
        '--bam', '-b',
        type=str,
        required=True,
        help="Path to the BAM file."
    )
    run_parser.add_argument(
        '--regions', '-r',
        type=str,
        required=True,
        help="Path to the regions file."
    )
    run_parser.add_argument(
        '--outdir', '-o',
        type=str,
        required=True,
        help="Directory to save the output images."
    )
    run_parser.add_argument(
        '--layers', '-l',
        type=str,
        nargs='+',
        required=True,
        help="List of layers to include in the generated images."
    )
    run_parser.add_argument(
        '--height', '-H',
        type=int,
        default=500,
        help="Maximum number of alignments (image height)."
    )
    run_parser.add_argument(
        '--min-cpg', '-c',
        type=int,
        default=1,
        help="Minimum number of CpGs within a read to include the read."
    )
    run_parser.add_argument(
        '--max-other-met', '-m',
        type=int,
        default=np.inf,
        help="Maximum number of non-CpG methylations allowed."
    )
    run_parser.add_argument(
        '--min-alignments', '-a',
        type=int,
        default=10,
        help="Minimum number of alignments for region to write image arrays for the region."
    )
    run_parser.add_argument(
        '--min-mapq', '-q',
        type=int,
        default=40,
        help="Minimum number of CpGs within a read to include the read."
    )
    run_parser.add_argument(
        '--single-ended',
        action='store_true',
        help="Assume single-ended reads (removes requirement for BAM to be sorted in any way."
    )
    run_parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress logging output."
    )

    # Subparser for 'layers' command
    layers_parser = subparsers.add_parser(
        "layers",
        help="List available layers."
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Draw image layers."
    )

    plot_parser.add_argument(
        'array_file',

        help='File name to draw.'
    )

    plot_parser.add_argument(
        'out_file',
        help='Output file. Image type will be infered from file name (I recommend "*.png").'
    )

    return argparser.parse_args()


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


def main():
    args = parse_args()

    if args.command == "run":
        del args.command
        if not check_layers(args.layers):
            exit(1)
        logger.remove()
        if not args.quiet:
            from jtmethtools.util import set_logger
            set_logger('INFO')
        del args.quiet
        run_image_gen(**vars(args))
    elif args.command == "layers":
        print_available_layers()
    elif args.command == 'plot':
        plot_an_image(args.array_file, args.out_file)



if __name__ == '__main__':
    main()