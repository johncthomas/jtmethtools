#!/usr/bin/env python
# A separate file from the functions that do anything as our HPC takes forever
#   to import large python packages and I want peoople to be able to run
#   `jtm-generate-images --help` and get a list of the command line options
#   quickly.

import argparse

def parse_args():
    argparser = argparse.ArgumentParser(
        description=(
            "Generate arrays from BAM files in specified regions, or view available "
            "layer methods.\n\nOutputs .tar.gz files that can be loaded in python with\n\t"
            "`array, metadata = jtmethtools.images.read_array(fn)`"
        )
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
        help="Path to the regions file. A BED or TSV with columns 'Name', 'Chrm', 'Start' & 'End'."
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
        help="List of layers to include in the generated images. Get list of valid layers with `jtm-generate-images layers`"
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
        default=5, # really big number so we don't have to import numpy for np.inf
        help="Maximum number of non-CpG methylations allowed in a read."
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
        help="Assume single-ended reads (reduces memory usage with unsorted BAMs)."
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

    return argparser.parse_args()

def parse_args_and_run():
    args = parse_args()
    # import after parsing args so it quickly runs and exits without proper args.
    from jtmethtools.generate_images import main
    main(args)

if __name__ == '__main__':
    parse_args_and_run()