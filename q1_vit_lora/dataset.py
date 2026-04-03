"""
CIFAR-100 dataset loading with ViT-compatible transforms.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS


# imagenet stats used by timm pretrained models
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(train=True):
    """Return train or val/test transforms resized to 224 for ViT."""
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomCrop(IMAGE_SIZE, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, val_split=0.1):
    """
    Returns train, val, test dataloaders for CIFAR-100.
    Val set is carved out of the training set.
    """
    train_full = datasets.CIFAR100(
        root=DATA_DIR, train=True, download=True,
        transform=get_transforms(train=True),
    )

    # split train into train + val
    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # val set uses eval transforms — wrap with a transform-overriding dataset
    val_set.dataset = datasets.CIFAR100(
        root=DATA_DIR, train=True, download=False,
        transform=get_transforms(train=False),
    )

    test_set = datasets.CIFAR100(
        root=DATA_DIR, train=False, download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
