# Another LMDB for PyTorch

This is a simple wrapper that lets you store a pytorch image dataset in an LMDB. This code wraps a `torch.utils.Dataset` that outputs image data and saves all images into a single database structure under the hood. In this case this an LMDB. There are a ton of variations of exactly this problem on github, but none were really what I wanted so I wrote another one.

#### What is different in this version?
* The wrapped dataset works like normal, it can be pickled for a dataloader, can have data augmentations and mirrors attributes of the wrapped dataset.
* No external dependencies on `pyarrow`, image data is directly written as `uint8` byte streams and directly read into tensor data.
* Data is saved uncompressed, but not in floats like in some other projects. This format is minimal and fast to read.
* Arbitrary image transformations such as resizing can be easily baked into the database.
* The LMDB can be written to a unique temporary file and cleaned up after deallocation.

### Installation
You can install this module via pip,
```
pip install torchlmdb
```
but it's really only a single file with a single class.

### Usage

Given an existing pytorch image dataset, for example
```
dataset = torchvision.datasets.CIFAR10(root="~/data", train=False)
```
simply wrap the dataset:
```
from torchlmdb import LMDBDataset
wrapped_dataset = LMDBDataset(dataset, name="val")
```
The wrapped dataset behaves like the original dataset, but stores the data in an LMDB database under the hood.


### Requirements:
The wrapped dataset needs to return pillow images that can be cast to `uint8` when `dataset.transform=None`. This is the case
for all of the torchvision image datasets, for example. Other transformations (for example random flips and normalization)
are ok and will be seamlessly applied to the LMDB output, so that random transforms are still random and not encoded into the db.
However, you can also encode transformations directly into the stored data. For example, calling `LMDBDataset` with `db_tranform=torchvision.transforms.Resize(16)` will store all data as 16x16 images in the database. Transforms like this are significantly more efficient than reading in a large image and downsampling it on the fly.
The `name` argument is user controlled and, when the database is not temporary, should uniquely identify this dataset when multiple databases exist in the root directory.

### Arguments:
```
LMDBDataset:

Args:
        dataset: The original image dataset (or any dataset that returns PIL data)
        root: Where to create the database
        name: A name for the newly created database
        can_create: Set to false to forbid database creation (For example in distrbuted training)
        temporary_db: Create the database only temporary and clean it up after deletion of this object
        db_transform: A torchvision.transform (or composition) to be applied during database creation.
        force_db_rebuild: Force a rebuilding of the database
        db_cfg: A struct of additional configuration options as described in the readme.
```


### Advanced Arguments:
A `db_cfg` can be handed as argument with additional arguments.
The default arguments can be imported as `LMDB_config` and are set to
```
    map_size = 1099511627776 * 2  # Linux can grow memory as needed.
    write_frequency = 65536  # how often to flush during database creation
    shuffle_while_writing = False  # Shuffle during DB creation
    db_channels_first = True # Write in CHW format if possible.
    rounds = 1  # Can write multiple rounds of the same dataset (for example with different augmentations).
    num_db_attempts = 10  # how many attempts to open the database
    max_readers = 128 # How many processes can read the database simultaneously.
    readahead = True  # beneficial for long sequential reads, disable when randomly accessing large DBs
    meminit = True
    max_spare_txns = 128
    access = "get"  # can be "cursor" or "get"
```

### References
This implementation of an LMDB interface in pyTorch is based on some older forks I made of
https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
and
https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py .
