"""LMBD dataset to wrap an existing dataset and "seamlessly" move all data into an LMDB."""

import os

import pickle
import copy
import warnings
import time
import datetime

import platform
import tempfile
import lmdb

import torch
import torchvision
import numpy as np
from PIL import Image

import logging

warnings.filterwarnings("ignore", "The given buffer is not writable", UserWarning)

log = logging.getLogger(__name__)


class LMDB_config:
    """Configuration options for the LMDB. These should be reasonable defaults.

    For large datasets with random access (for example ImageNet trained with shuffle=True), try setting readahead=False.
    """

    # Writing:
    map_size = 1099511627776 * 2  # Linux can grow memory as needed.
    write_frequency = 65536  # how often to flush during database creation
    shuffle_while_writing = False  # Shuffle during DB creation
    db_channels_first = True
    rounds = 1  # Can write multiple rounds of the same dataset (for example with different augmentations).

    # reading:
    num_db_attempts = 10  # how many attempts to open the database
    max_readers = 128
    readahead = True  # this should be beneficial for long sequential reads, disable when randomly accessing large DBs
    meminit = True
    max_spare_txns = 128

    # access
    access = "get"  # cursor or get


to_tensor_transforms = (torchvision.transforms.ToTensor, torchvision.transforms.PILToTensor)


class LMDBDataset(torch.utils.data.Dataset):
    """Implement LMDB caching and access for image-based pytorch dataset.

    Use this class to wrap an existing dataset. After wrapping, the wrapped dataset behaves like the original dataset,
    but the underlying images are moved into an LMDB (Lightning Memory-Mapped Database Manager).

    When temporary_db=False and force_db_rebuild=False, the second invocation of a database with the same name at the
    same root location will read the existing database instead of building another one. It is the user's responsibility
    to make sure that the existing database with the same name contains the same data.

    Args:
            dataset: The original image dataset (or any dataset that returns PIL data)
            root: Where to create the database
            name: A name for the newly created database
            can_create: Set to false to forbid database creation (For example in distrbuted training)
            temporary_db: Create the database only temporary and clean it up after deletion of this object
            db_transform: A torchvision.transform (or composition) to be applied during database creation.
            force_db_rebuild: Force a rebuilding of the database
            db_cfg: A struct of additional configuration options as described in the readme.

    This object can be pickled and hence, for example, multiple workers can be used in a PyTorch dataloader.

    """

    def __init__(
        self,
        dataset,
        root="~/data",
        name="",
        can_create=True,
        temporary_db=False,
        db_transform=torchvision.transforms.PILToTensor(),
        force_db_rebuild=False,
        db_cfg=LMDB_config,
    ):
        """Initialize with given pytorch dataset, figure out transforms to apply and read or create the database."""
        self.dataset = dataset
        self.db_cfg = db_cfg

        self.live_transform = self._check_transforms(dataset.transform)
        self.path, self.handle = self._choose_lmdb_path(dataset, root, temporary_db, name, db_cfg)
        if can_create:
            if force_db_rebuild and not temporary_db:
                if os.path.isfile(self.path):
                    os.remove(self.path)
                    os.remove(self.path + "-lock")

            # Load or create database
            if os.path.isfile(self.path) and not temporary_db:
                log.info(f"Reusing cached database at {self.path}.")
            else:
                log.info(f"Creating database at {self.path}. This may take some time ...")
                checksum = self._create_database(dataset, self.path, db_transform, name, db_cfg)

        # Setup database
        self.access = db_cfg.access
        for attempt in range(db_cfg.num_db_attempts):
            self.db = lmdb.open(
                self.path,
                subdir=False,
                max_readers=db_cfg.max_readers,
                readonly=True,
                lock=False,
                readahead=db_cfg.readahead,
                meminit=db_cfg.meminit,
                max_spare_txns=db_cfg.max_spare_txns,
            )
            try:
                with self.db.begin(write=False) as txn:
                    self.length = pickle.loads(txn.get(b"__len__"))
                    self.keys = pickle.loads(txn.get(b"__keys__"))
                    self.labels = pickle.loads(txn.get(b"__labels__"))
                    self.shape = pickle.loads(txn.get(b"__shape__"))
                break
            except TypeError:
                warnings.warn(
                    f"The provided LMDB dataset at {self.path} is unfinished or damaged. Waiting and retrying."
                )
                time.sleep(13)
        else:
            raise ValueError(f"Database at path {self.path} damaged and could not be loaded after repeated attempts.")

        if self.access == "cursor":
            self._init_cursor()

    def _check_transforms(self, transform):
        """Decides whether to store data in HWC or CHW format.

        If the original dataset.transform does not contain pillow transformations, then the dataset will be stored as
        CHW directly, which saves a transpose operation during data loading.
        If the original dataset.transform does contain a pillow transform, the dataset will be stored as HWC.
        """
        live_transform = copy.deepcopy(transform)
        if live_transform is not None:
            # ducktype whether live_transform is a composition of transforms
            try:
                contains_only_tensor_transforms = isinstance(live_transform.transforms[0], to_tensor_transforms)
                if contains_only_tensor_transforms:
                    slive_transform.transforms.pop(0)
            except (TypeError, AttributeError):
                contains_only_tensor_transforms = isinstance(live_transform, to_tensor_transforms)
                if contains_only_tensor_transforms:
                    live_transform = None
            if contains_only_tensor_transforms:
                self.skip_pillow = True
                self.db_cfg.db_channels_first = True
            else:
                self.skip_pillow = False
                self.db_cfg.db_channels_first = False
        else:
            self.skip_pillow = True
            self.db_cfg.db_channels_first = True
        return live_transform

    def __getstate__(self):
        """Overwritten to allow pickling."""
        state = self.__dict__
        state["db"] = None
        if self.access == "cursor":
            state["_txn"] = None
            state["cursor"] = None
        return state

    def __setstate__(self, state):
        """Overwritten to allow pickling. The db handle will be regenerated after pickling."""
        self.__dict__ = state
        # Regenerate db handle after pickling:
        self.db = lmdb.open(
            self.path,
            subdir=False,
            max_readers=self.db_cfg.max_readers,
            readonly=True,
            lock=False,
            readahead=self.db_cfg.readahead,
            meminit=self.db_cfg.meminit,
            max_spare_txns=self.db_cfg.max_spare_txns,
        )
        if self.access == "cursor":
            self._init_cursor()

    def _init_cursor(self):
        """Initialize cursor position. This is an optional way to index the LMDB."""
        self._txn = self.db.begin(write=False)
        self.cursor = self._txn.cursor()
        self.cursor.first()
        self.internal_index = 0

    def __getattr__(self, name):
        """This wrapper defaults to attributes from the source dataset instead of raising an AttributeError.

        This is useful for a variety of simple attributes (i.e. dataset metadata, dataset class names), but should be
        handled with care.
        """
        return getattr(self.dataset, name)

    def __len__(self):
        """Draw length from the LMDB key."""
        return self.length

    def __getitem__(self, index):
        """Get from database. This is either unordered ("get" access) or cursor access."""

        if self.access == "cursor":
            index_key = "{}".format(index).encode("ascii")
            if index_key != self.cursor.key():
                self.cursor.set_key(index_key)

            byteflow = self.cursor.value()
            self.cursor.next()
        else:
            with self.db.begin(write=False) as txn:
                byteflow = txn.get(self.keys[index])

        # crime, but ok - we just disabled the warning...:
        # Tested this and the LMDB cannot be corrupted this way, even though byteflow is technically non-writeable
        data_block = torch.frombuffer(byteflow, dtype=torch.uint8).view(self.shape)

        if not self.skip_pillow:
            img = Image.fromarray(data_block.numpy())
        else:
            img = data_block.to(torch.float) / 255
        if self.live_transform is not None:
            img = self.live_transform(img)

        # load label
        label = self.labels[index]

        return img, label

    @staticmethod
    def _choose_lmdb_path(raw_dataset, root, temporary_db, name, db_cfg=LMDB_config):
        """The LMDB is named like CIFAR10_CHOSENNAME_50000_HWC.lmdb if temporary_db=False.

        If temporary_db=True, pythons tempfile module will chose a name and ignore the "name" argument.
        """
        root = os.path.expanduser(root)
        os.makedirs(root, exist_ok=True)
        if os.path.isfile(root):
            raise ValueError("LMDB path must lead to a folder containing the databases, not a file.")

        if not temporary_db:
            layout = "_CHW" if db_cfg.db_channels_first else "HWC"
            path = os.path.join(root, f"{type(raw_dataset).__name__}_{name}_{len(raw_dataset)}_{layout}.lmdb")
            handle = None
        else:
            handle = tempfile.NamedTemporaryFile(dir=root)
            path = handle.name

        return path, handle

    @staticmethod
    def _create_database(dataset, database_path, db_transform, name="train", db_cfg=LMDB_config):
        """Create the LMDB database from the given pytorch dataset.

        Each database entry is a blob of uint8 data describing each image.
        """
        data_transform = copy.deepcopy(dataset.transform)

        dataset.transform = db_transform
        if platform.system() != "Linux" and db_cfg.map_size == 1099511627776 * 2:
            raise ValueError("Provide a reasonable default map_size for your operating system.")
        else:
            map_size = db_cfg.map_size
        db = lmdb.open(
            database_path,
            subdir=False,
            map_size=map_size,
            readonly=False,
            meminit=db_cfg.meminit,
            writemap=True,
            map_async=True,
        )
        txn = db.begin(write=True)

        num_workers = min(16, torch.get_num_threads())
        batch_size = min(len(dataset) // max(num_workers, 1), 512)
        shape = dataset[0][0].shape

        cacheloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=db_cfg.shuffle_while_writing,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True if ((num_workers > 0) and (db_cfg.rounds > 1)) else None,
        )

        idx = 0
        timestamp = time.time()
        labels = []
        for round in range(db_cfg.rounds):
            for batch_idx, (img_batch, label_batch) in enumerate(cacheloader):
                # Run data transformations in (multiprocessed) batches
                for img, label in zip(img_batch, label_batch):
                    # But we have to write sequentially anyway
                    if img.shape != shape:
                        raise ValueError("All entries need to be cropped/resized to the same shape.")
                    labels.append(label.item())
                    # serialize
                    if db_cfg.db_channels_first:
                        byteflow = np.asarray(img.numpy(), dtype=np.uint8).tobytes()
                    else:
                        byteflow = np.asarray(img.permute(1, 2, 0).numpy(), dtype=np.uint8).tobytes()
                    txn.put("{}".format(idx).encode("ascii"), byteflow)
                    idx += 1

                    if idx % db_cfg.write_frequency == 0:
                        time_taken = (time.time() - timestamp) / db_cfg.write_frequency
                        estimated_finish = str(datetime.timedelta(seconds=time_taken * len(dataset) * db_cfg.rounds))
                        timestamp = time.time()

                        txn.commit()
                        txn = db.begin(write=True)
                        log.info(f"[{idx} / {len(dataset) * db_cfg.rounds}] Estimated total time: {estimated_finish}")

        # finalize dataset
        txn.commit()
        keys = ["{}".format(k).encode("ascii") for k in range(idx)]
        shape = img.shape if db_cfg.db_channels_first else img.permute(1, 2, 0).shape
        with db.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(keys))
            txn.put(b"__labels__", pickle.dumps(labels))
            txn.put(b"__len__", pickle.dumps(len(keys)))
            txn.put(b"__shape__", pickle.dumps(shape))
        log.info(f"Database written successfully with {len(keys)} entries of shape {shape}.")
        dataset.transform = data_transform
