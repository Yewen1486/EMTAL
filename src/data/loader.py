#!/usr/bin/env python3

"""Data loader."""
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler




def _construct_loader(split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = 'vtab-cifar(num_classes=100)'

    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        from .datasets.tf_dataset import TFDataset
        dataset = TFDataset(split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader( ):
    """Train loader wrapper."""
    drop_last = True
    return _construct_loader(
        split="train",
        batch_size=int(64 / 2),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_trainval_loader():
    """Train loader wrapper."""
    drop_last = True
    return _construct_loader(
        split="trainval",
        batch_size=32,
        shuffle=True,
        drop_last=drop_last,
    )


def construct_test_loader():
    """Test loader wrapper."""
    return _construct_loader(
        split="test",
        batch_size=32,
        shuffle=False,
        drop_last=False,
    )


def construct_val_loader(batch_size=None):
    bs = 32
    """Validation loader wrapper."""
    return _construct_loader(
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
