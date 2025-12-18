import pysam
from pysam import AlignedSegment
from typing import Collection, Iterable, Tuple
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from jtmethtools.alignments import iter_bam_segments
from loguru import logger
from datargs import argsclass, parse
from dataclasses import field

BamReader = Iterable[tuple[AlignedSegment, AlignedSegment|None]]

def get_nreads_from_bai(bam_path:Path) -> int:
    """Get number of reads in a BAM file using its BAI index.

    Args:
        bam_path: Path to the BAM file.
    """
    bamfile = pysam.AlignmentFile(str(bam_path), "rb")
    n_reads = bamfile.mapped + bamfile.unmapped
    bamfile.close()
    return n_reads


def write_sampled_alignments(
        bam_reader:BamReader,
        bam_writer:pysam.AlignmentFile,
        sampled_indices:Collection[int],
        replacement:bool,
        output_paired:bool,
        new_qname_prefix:str=None
):
    """Write sampled alignments from a BAM reader to a BAM .

    If not replacement, assumes sampled indicies are unique.

    If not replacement, renames reads' query names to "{new_qname_prefix}_{i}"
    where i is the index in the original BAM.

    If replacement, query names renamed to "{new_qname_prefix}_{i}_{j}"
    where i is the index in the original BAM and j is the number of times
    this read has been written so far.

    If new_qname_prefix is None, original read names are used. When
    replacement==True, numbers are added to ."""

    # if new_qname_prefix is None, original read names are used
    prefix = new_qname_prefix

    if not replacement:
        sampled_indices = set(sampled_indices)
        for i, (a1, a2) in enumerate(bam_reader):
            if i not in sampled_indices:
                continue
            # rename if prefix set, otherwise no modification needed.
            if new_qname_prefix is not None:
                a1.query_name = f"{prefix}_{i}"
                try:
                    a2.query_name = f"{prefix}_{i}"
                except AttributeError:
                    pass
            bam_writer.write(a1)
            if output_paired:
                if a2 is not None:
                    bam_writer.write(a2)
                else:
                    bam_writer.write(a1)

    elif replacement:
        values, counts = np.unique(np.asarray(sampled_indices, dtype=int), return_counts=True)
        counts_map = {int(v): int(c) for v, c in zip(values, counts)}

        for i, (a1, a2) in enumerate(bam_reader):
            if i not in counts_map:
                continue
            count = counts_map[i]

            for j in range(count):
                if new_qname_prefix is None:
                    prefix = a1.query_name

                a1.query_name = f"{prefix}_{i}_{j}"
                try:
                    a2.query_name = f"{prefix}_{i}_{j}"
                except AttributeError:
                    pass
                bam_writer.write(a1)
                if output_paired:
                    if a2 is not None:
                        bam_writer.write(a2)
                    else:
                        bam_writer.write(a1)


def create_synthetic_bam(
        inputs:Iterable[tuple[str|Path, float, bool, bool]],
        total_reads:int,
        output_bam:Path,
        output_paired:bool,
        seed=None,
        use_filestem_as_prefix:bool=True
) -> None:
    """Write a BAM made of a mixture of input BAMs according to specified proportions.

    Args:
        inputs: Gives (input_bam, proportion, replacement, paired) for each input BAM.
        total_reads: Total number of reads in the output BAM.
        output_bam: Path to the output BAM file.
        output_paired: When True single ended inputs are written twice. When False
            only the first read of a pair is written.
        use_filestem_as_prefix: If True, use the stem of BAM file name as
            the prefix for output read names. If False, use original read names.
        seed: Optional random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    assert all([len(x) == 4 for x in inputs]), "Each input must be a tuple of (input_bam, proportion, replacement, paired)."

    input_bams = [Path(f[0]) for f in inputs]
    input_proportions = [f[1] for f in inputs]
    input_replacement = [f[2] for f in inputs]
    input_paired = [f[3] for f in inputs]

    assert all([f.exists() for f in input_bams]), "One or more input BAM files do not exist."

    input_proportions = np.array(input_proportions, dtype=float)/sum(input_proportions)

    # Prepare output BAM file
    template = pysam.AlignmentFile(str(input_bams[0]), "rb")
    try:
        with pysam.AlignmentFile(str(output_bam), "wb", template=template) as out_bam:
            for fn, prop, replace, paired in zip(
                    input_bams, input_proportions, input_replacement, input_paired
            ):
                reader = iter_bam_segments(fn, paired_end=paired)
                # count reads in BAM
                in_reads = -1
                for in_reads, _ in enumerate(reader):
                    pass
                if in_reads == -1:
                    raise ValueError(f"No reads found in BAM file {fn}.")
                in_reads += 1  # zero-based to count
                out_reads = int(round(total_reads * input_proportions[input_bams.index(fn)], ndigits=0))

                if not replace and (out_reads > in_reads):
                    raise ValueError(
                        f"Cannot sample {out_reads} reads without replacement "
                        f"from BAM file {fn} containing only {in_reads} reads."
                    )

                # sample read indices
                sampled_indices = np.random.choice(
                    in_reads,
                    size=out_reads,
                    replace=replace
                )
                reader = iter_bam_segments(fn, paired_end=paired)
                prfx = fn.stem if use_filestem_as_prefix else None
                write_sampled_alignments(
                    reader,
                    out_bam,
                    sampled_indices,
                    replacement=replace,
                    new_qname_prefix=prfx,
                    output_paired=output_paired,
                )
    finally:
        template.close()


def ttest():
    import pytest
    samstr = """\
@HD	VN:1.0	SO:none
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
"""

    samstr2 = """\
@HD	VN:1.0	SO:none
@SQ	SN:1	LN:248956422
mapq20Unorder2	18	1	100	20	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
unmethRev2	18	1	100	42	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
methRev2	18	1	101	32	10M	*	0	10	CCCCCCCCCC	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:HHHHHHHHHH
unmethFor2	2	1	102	27	10M	*	0	10	TTTTTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:zzzzzzzzzz
methFor2	2	1	103	22	10M	*	0	10	CCCCCCCCCC	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:ZZZZZZZZZZ
allA2	2	1	104	17	10M	*	0	10	AAAAAAAAAA	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allC2	2	1	105	12	10M	*	0	10	CCCCCCCCCC	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allG2	2	1	106	12	10M	*	0	10	GGGGGGGGGG	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
allT2	2	1	107	12	10M	*	0	10	TTTTTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..........
oneCpG2	2	1	107	12	10M	*	0	10	TTCGTTTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..z.......
twoCpG2	2	1	107	12	10M	*	0	10	TTCGCGTTTT	ABCDEFGHIJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:..z.z.....
mapq192	18	1	100	19	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
mapq202	18	1	100	20	10M	*	0	10	TTTTTTTTTT	JJJJJJJJJJ	NM:i:tag0\tMD:Z:tag1\tXM:Z:hhhhhhhhhh
oneCHH2\t2\t1\t104\t22\t10M\t*\t0\t10\tTTCTTTTTTT\tABCDEFGHIJ\tNM:i:tag0\tMD:Z:tag1\tXM:Z:..H.......
twoCHH2\t2\t1\t104\t22\t10M\t*\t0\t10\tTTCTTCTTTT\tABCDEFGHIJ\tNM:i:tag0\tMD:Z:tag1\tXM:Z:..H..H....
"""

    tmp_path = Path("/home/jcthomas/tmp/synthetic_sample_test/")
    tmp_path.mkdir(parents=True, exist_ok=True)

    sam_path_t = "test_synthetic{}.sam"
    with open(tmp_path/sam_path_t.format(1), 'w') as f:
        f.write(samstr)
    with open(tmp_path/sam_path_t.format(2), 'w') as f:
        f.write(samstr2)


    seed = 19810613
    sam1 = Path(tmp_path/"test_synthetic1.sam")
    sam2 = Path(tmp_path/"test_synthetic2.sam")

    # require the SAMs to exist
    if not sam1.exists() or not sam2.exists():
        raise FileNotFoundError("Ensure `test_synthetic1.sam` and `test_synthetic2.sam` exist in the cwd.")

    inputs_equal = [
        (sam1, 0.3, True, False),
        (sam2, 0.7, True, False),
    ]

    # 1) Replacement: total 30 (should produce 30 output alignments)
    create_synthetic_bam(
        inputs=inputs_equal,
        total_reads=30,
        output_bam=Path(tmp_path/"out_repl_30.bam"),
        seed=seed,
        output_paired=False,
    )

    # 2) Without replacement: total 10 (should produce 10 output alignments)
    inputs_no_replace = [
        (sam1, 0.3, False, False),
        (sam2, 0.7, False, False),
    ]
    create_synthetic_bam(
        inputs=inputs_no_replace,
        total_reads=10,
        output_bam=Path(tmp_path/"out_no_repl_10.bam"),
        seed=seed,
        output_paired=False,
    )

    # 3) Trigger the ValueError for not-enough-reads (catch and write an .err file)
    with pytest.raises(ValueError, match=r"^Cannot sample \d+(\.\d+)? reads without replacement"):
        create_synthetic_bam(
            inputs=inputs_no_replace,
            total_reads=100,  # should be too many
            output_bam=Path(tmp_path/"out_should_error.bam"),
            seed=seed,
            output_paired=False,
        )


    # 4) With/without replacement but with new_qname_prefix forced to None
    # replacement case (prefix None)
    inputs_mixed_replace = [
        (sam1, 0.3, True, False),
        (sam2, 0.7, True, False),
    ]
    create_synthetic_bam(
        inputs=inputs_mixed_replace,
        total_reads=10,
        output_bam=Path(tmp_path/"out_no_prefix_repl.bam"),
        seed=seed,
        use_filestem_as_prefix=False,
        output_paired=False,
    )

    # no-replacement case (prefix None)
    inputs_mixed_norepl = [
        (sam1, 0.3, False, False),
        (sam2, 0.7, False, False),
    ]
    create_synthetic_bam(
        inputs=inputs_mixed_norepl,
        total_reads=10,
        output_bam=Path(tmp_path/"out_no_prefix_no_repl.bam"),
        seed=seed,
        use_filestem_as_prefix=False,
        output_paired=False,
    )


@argsclass(
    description="""\
Write a BAM that is a synthetic mixture of input BAM files according to specified proportions.
"""
)
class SyntheticMixtureArgs:
    input_files: list[str] = field(metadata=dict(
        required=True,

        help="Input BAM files with proportions and replacement flags, in the format: "
             "-i input1.bam,0.3,T,F -i input2.bam,0.7,F,T ... Proportions are normalised to "
             "sum to 1. 'T'/'F' indicate whether sampling is with replacement and whether the BAM is paired-end.",
        nargs='+',
        aliases=['-i'],
    ))
    total_reads: int = field(metadata=dict(
        required=True,
        help="Total number of reads in the output BAM file.",
        aliases=['-n'],
    ))
    out: Path = field(metadata=dict(
        required=True,
        help="Output BAM file path.",
        aliases=['-o'],
    ))
    pe: bool = field(metadata=dict(

        help="Set if the *output* BAM should be paired-end. Either --pe or --se must be set. "
             "Single-ended inputs will be duplicated to make pairs if --pe is set. "
             "If --se is set, paired-ended inputs will have the first read written only.",
        aliases=['--paired', '--pe'],
    ), default=False,)
    se: bool = field(metadata=dict(
        help="Set if the *output* BAM should be single-end.",
        aliases=['--single', '--se'],
    ), default=False,)
    seed: int = field(metadata=dict(
        help="Random seed for reproducibility.",
    ), default=None,)

def cli(args: SyntheticMixtureArgs):

    if args.pe == args.se:
        raise ValueError("Either --pe or --single must be set, but not both.")

    # split out the inputs
    inputs = []
    for input_str in args.input_files:
        parts = input_str.split(',')
        if len(parts) != 4:
            raise ValueError(
                f"Input specification '{input_str}' is invalid. "
                f"Expected format: input.bam,proportion,replacement,paired"
            )
        input_bam = parts[0]
        proportion = float(parts[1])
        replacement = parts[2].strip().upper().startswith('T')
        paired = parts[3].strip().upper() == 'T'
        inputs.append((input_bam, proportion, replacement, paired))

    create_synthetic_bam(
        inputs=inputs,
        total_reads=args.total_reads,
        output_bam=args.out,
        seed=args.seed,
        output_paired=args.pe,
    )

if __name__ == "__main__":
    clargs = parse(SyntheticMixtureArgs)
    cli(clargs)