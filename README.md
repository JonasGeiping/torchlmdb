# Another LMDB for PyTorch

This is a wrapper for PyTorch datasets that contain images (more precisely, that return `PIL.image` objects when no `dataset.transform` is applied.).



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

### References
This implementation of an LMDB interface in pyTorch is based on some older forks I made of
https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
and
https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py .
for a previous project, https://github.com/JonasGeiping/fullbatch .
