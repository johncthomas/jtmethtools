
Recipe to get a list of bools indicating methylation or not. 
```python
    from alignments import *
    bamfn = 'file.bam'
    regfn = 'file.bed'
    for alignment in iter_bam(bamfn, (0, 6), paired_end=False):
        meth_by_locus = alignment.locus_methylation
        print(alignment.a.query_name)
        print(alignment.metstr)
        print(meth_by_locus.values())

    # filter regions by bed and absence of CHH etc
    # pysam AlignedSegment objects are alignment.a (and alignment.a2 if paired)
    #   so use them for further filtering e.g. alignment.a.mapping_quality > 20
    regions = Regions.from_bed(regfn)
    count = 0
    for alignment in iter_bam(bamfn, (0, 10000), paired_end=True):

        if alignment.hit_regions(regions) and alignment.no_non_cpg():
            count += 1
    print(count)
```