
Recipe to get a list of bools indicating methylation or not
```python
    from jtmethtools import iter_bam
    bamfn = '/home/jcthomas/data/canary/sorted_qname/CMDL19003169_1_val_1_bismark_bt2_pe.deduplicated.bam'

    for alignment in iter_bam(bamfn, (0, 6), paired_end=False):
        meth_by_locus = alignment.locus_methylation
        print(alignment.a.query_name)
        print(alignment.metstr)
        print(meth_by_locus.values())
```