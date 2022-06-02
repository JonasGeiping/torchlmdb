import torch
import torchvision
from torchlmdb import LMDBDataset

import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

dataset = torchvision.datasets.CIFAR10(root="~/data", train=False, transform=torchvision.transforms.ToTensor())
print(dataset)
lmdb_dataset = LMDBDataset(dataset, name="cifar_val", force_db_rebuild=True)
print(lmdb_dataset)
print(torch.all(torch.isclose(lmdb_dataset[0][0], dataset[0][0])))


dataset2 = torchvision.datasets.CIFAR10(
    root="~/data",
    train=False,
    transform=torchvision.transforms.Compose([torchvision.transforms.Resize(64), torchvision.transforms.ToTensor()]),
)
print(dataset2)
lmdb_dataset2 = LMDBDataset(dataset2, name="cifar_val_dataaug", force_db_rebuild=True)
print(lmdb_dataset2)
print(torch.all(torch.isclose(lmdb_dataset2[0][0], dataset2[0][0])))


dataset = torchvision.datasets.CIFAR10(root="~/data", train=False, transform=torchvision.transforms.ToTensor())
lmdb_dataset = LMDBDataset(
    dataset,
    name="cifar_val_dataaug",
    force_db_rebuild=True,
    db_transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize(64), torchvision.transforms.PILToTensor()]
    ),
)
print(torch.all(torch.isclose(lmdb_dataset[0][0], dataset2[0][0])))

lmdb_dataset = LMDBDataset(dataset, name="cifar_val", force_db_rebuild=True, temporary_db=False)
lmdb_dataset = LMDBDataset(dataset, name="cifar_val", force_db_rebuild=False, temporary_db=False)
lmdb_dataset = LMDBDataset(dataset, name="cifar_val", force_db_rebuild=True, temporary_db=True)

dataset = torchvision.datasets.CIFAR10(
    root="~/data",
    train=False,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1, 0)),
        ]
    ),
)
lmdb_dataset = LMDBDataset(dataset, name="cifar_val", force_db_rebuild=True, temporary_db=True)
