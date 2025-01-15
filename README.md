Methylation data tools. Arrow based tables for efficiently storing and processing Bismark BAMs. Module for producing pile-up images of regions for CNNs.

# Images for CNN
2D pileups, as binary arrays with values between 0 & 1 representing different sequence features such as methylation state, 
mapping quality and nucleotide sequence.

Paired-end BAMs should be sorted by query name (preferably) or coordinate (may take more memory). Unsorted BAMs
will probably work but use a lot of memory.

## Generation
After installation use `jtm-generate-images run --help` for arguments. Available layers (that can then be passed
to the `--layer` option) can be printed using `jtm-generate-images layers`. `run` produces gzipped TAR files that
contain the binary array and a metadata JSON file, specifying the shape of the array and other things.

`jtm-generate-images` invokes the script `generate_images.py`. 

## Loading arrays
```python
import jtmethtools as jtm
fn = 'image.region_name.layer.tar.gz'
array, metadata = jtm.images.read_array(fn)
```

## plot an array
```python
plt.imshow(array, interpolation='nearest', cmap='gray')
```
