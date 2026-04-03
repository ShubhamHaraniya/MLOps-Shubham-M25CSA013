"""
Configuration for Q2 adversarial attacks.
Paths resolve relative to this file so code runs from any directory.
"""

import os
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
_ROOT      = _HERE.parent

DATA_DIR   = str(_ROOT / "data")
WEIGHT_DIR = str(_ROOT / "weights" / "q2_all")
RESULT_DIR = str(_ROOT / "results" / "q2")

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mlops-assignment5")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY",  None)

# ResNet-18 clean training
NUM_CLASSES_CIFAR10 = 10
TRAIN_EPOCHS        = 30
BATCH_SIZE          = 128
LR_RESNET           = 0.1
MOMENTUM            = 0.9
WEIGHT_DECAY        = 5e-4
NUM_WORKERS         = 4

# FGSM epsilon values
FGSM_EPSILONS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]

# PGD attack params
PGD_EPS  = 8 / 255
PGD_STEP = 2 / 255
PGD_ITER = 40

# BIM attack params
BIM_EPS  = 8 / 255
BIM_STEP = 2 / 255
BIM_ITER = 40

# Detector training
DETECTOR_EPOCHS = 20
DETECTOR_LR     = 1e-4
