"""
CIFAR-10 data loading for Q2 adversarial experiments.
Returns both PyTorch DataLoaders and raw numpy arrays (required by ART).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS

# CIFAR-10 stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Return (train_loader, test_loader) for CIFAR-10."""
    train_set = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True,
        transform=get_transforms(train=True),
    )
    test_set = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True,
        transform=get_transforms(train=False),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader


def get_numpy_data(n_test=1000):
    """
    Return numpy arrays for ART (scaled to [0,1]).
    ART works with unnormalized data; normalization goes inside the model wrapper.
    """
    test_set = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True,
        transform=transforms.ToTensor(),   # just [0,1], no normalisation
    )
    loader = DataLoader(test_set, batch_size=n_test, shuffle=False)
    images, labels = next(iter(loader))
    return images.numpy(), labels.numpy()


def get_numpy_train_data(n_train=5000):
    """Return subset of training data as numpy arrays (unnormalized)."""
    train_set = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(train_set, batch_size=n_train, shuffle=True)
    images, labels = next(iter(loader))
    return images.numpy(), labels.numpy()
