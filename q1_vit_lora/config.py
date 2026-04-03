"""
Hyperparameter configs for Q1 ViT-S LoRA experiments on CIFAR-100.
Paths are resolved relative to this file so the code runs from any directory.
"""

import os
from pathlib import Path

# project root = two levels up from this file (q1_vit_lora/config.py)
_HERE        = Path(__file__).resolve().parent
_ROOT        = _HERE.parent

DATA_DIR   = str(_ROOT / "data")
WEIGHT_DIR = str(_ROOT / "weights" / "q1_best")
RESULT_DIR = str(_ROOT / "results" / "q1")

# wandb
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mlops-assignment5")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY",  None)

# training defaults
NUM_CLASSES  = 100
EPOCHS       = 10
BATCH_SIZE   = 64
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
IMAGE_SIZE   = 224
MODEL_NAME   = "vit_small_patch16_224"

# LoRA experiment grid
# exp 0 = no LoRA baseline, exps 1-9 = rank x alpha combos
LORA_EXPERIMENTS = [
    # (experiment_id, use_lora, rank, alpha, dropout)
    (0, False, None, None, None),
    (1, True,  2,    2,    0.1),
    (2, True,  2,    4,    0.1),
    (3, True,  2,    8,    0.1),
    (4, True,  4,    2,    0.1),
    (5, True,  4,    4,    0.1),
    (6, True,  4,    8,    0.1),
    (7, True,  8,    2,    0.1),
    (8, True,  8,    4,    0.1),
    (9, True,  8,    8,    0.1),
]


def get_experiment_config(exp_id):
    for eid, use_lora, rank, alpha, dropout in LORA_EXPERIMENTS:
        if eid == exp_id:
            return {
                "exp_id": eid, "use_lora": use_lora,
                "rank": rank, "alpha": alpha, "dropout": dropout,
                "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                "lr": LR, "weight_decay": WEIGHT_DECAY,
                "model_name": MODEL_NAME, "image_size": IMAGE_SIZE,
                "num_classes": NUM_CLASSES,
            }
    raise ValueError(f"Unknown experiment id: {exp_id}")


def get_run_name(cfg):
    if not cfg["use_lora"]:
        return "baseline_no_lora"
    return f"lora_r{cfg['rank']}_a{cfg['alpha']}_d{cfg['dropout']}"
