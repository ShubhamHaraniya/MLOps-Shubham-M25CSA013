"""
ResNet-18 (CIFAR-10 classifier) and ResNet-34 (adversarial detector).
Modified first conv for 32x32 CIFAR images.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights


class NormalizeLayer(nn.Module):
    """Normalize input images. Lets ART work with raw [0,1] inputs."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def build_resnet18_cifar10():
    """
    ResNet-18 adapted for CIFAR-10 (32×32 images):
    - Replace 7×7 conv with 3×3 conv, remove max-pool.
    - Includes internal normalisation so ART can pass raw [0,1] images.
    """
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

    backbone = models.resnet18(weights=None, num_classes=10)
    # adapt for 32x32
    backbone.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()

    class ResNet18CIFAR10(nn.Module):
        def __init__(self):
            super().__init__()
            self.normalize = NormalizeLayer(CIFAR10_MEAN, CIFAR10_STD)
            self.backbone  = backbone

        def forward(self, x):
            return self.backbone(self.normalize(x))

    return ResNet18CIFAR10()


def build_resnet34_detector(num_classes=2):
    """
    ResNet-34 for binary adversarial detection (clean=0, adversarial=1).
    Adapted for CIFAR-10 32×32 inputs with internal normalisation.
    """
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

    backbone = models.resnet34(weights=None, num_classes=num_classes)
    backbone.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()

    class ResNet34Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.normalize = NormalizeLayer(CIFAR10_MEAN, CIFAR10_STD)
            self.backbone  = backbone

        def forward(self, x):
            return self.backbone(self.normalize(x))

    return ResNet34Detector()
