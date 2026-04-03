"""
Push best ViT-S LoRA model weights to HuggingFace Hub.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import torch
from huggingface_hub import HfApi, create_repo

from config import WEIGHT_DIR, WANDB_PROJECT


HF_REPO_ID = os.environ.get("HF_REPO_ID", "ShubhamHaraniya/vit-small-cifar100-lora")
HF_TOKEN   = os.environ.get("HF_TOKEN", None)


MODEL_CARD = """---
license: mit
tags:
  - image-classification
  - vision-transformer
  - lora
  - peft
  - cifar100
datasets:
  - cifar100
---

# ViT-Small CIFAR-100 (LoRA Fine-tuned)

This model is a `vit_small_patch16_224` from [timm](https://github.com/huggingface/pytorch-image-models),
fine-tuned on **CIFAR-100** using **LoRA (Low-Rank Adaptation)** via the
[PEFT](https://github.com/huggingface/peft) library.

## Training Details

- **Base model**: `vit_small_patch16_224` (ImageNet pretrained)
- **Dataset**: CIFAR-100 (100 classes)
- **Method**: LoRA injected into attention `qkv` layers
- **WandB project**: `{wandb_project}`

## Usage

```python
import torch
import timm
from peft import LoraConfig, get_peft_model

model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=100)
lora_config = LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=["qkv"],
                         lora_dropout=0.1, bias="none", modules_to_save=["head"])
model = get_peft_model(model, lora_config)

ckpt = torch.load("pytorch_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```
""".format(wandb_project=WANDB_PROJECT)


def push_best_model(ckpt_path=None):
    """Upload best checkpoint + model card to HuggingFace Hub."""
    api = HfApi()

    # find checkpoint if not specified
    if ckpt_path is None:
        # prefer optuna best, then any exp checkpoint
        optuna_ckpt = os.path.join(WEIGHT_DIR, "optuna_best_model.pt")
        if os.path.exists(optuna_ckpt):
            ckpt_path = optuna_ckpt
        else:
            candidates = sorted(glob.glob(os.path.join(WEIGHT_DIR, "*.pt")))
            if not candidates:
                raise FileNotFoundError(f"No .pt files found in {WEIGHT_DIR}")
            # pick the one with highest val_acc stored in checkpoint
            best_acc, best_path = -1, None
            for p in candidates:
                ckpt = torch.load(p, map_location="cpu")
                if ckpt.get("val_acc", 0) > best_acc:
                    best_acc = ckpt["val_acc"]
                    best_path = p
            ckpt_path = best_path

    print(f"Uploading: {ckpt_path} => {HF_REPO_ID}")

    # create repo if needed
    create_repo(HF_REPO_ID, token=HF_TOKEN, exist_ok=True, private=False)

    # upload model weights
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo="pytorch_model.pt",
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
    )

    # upload model card
    card_path = "/tmp/README.md"
    with open(card_path, "w") as f:
        f.write(MODEL_CARD)
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
    )

    print(f"Model pushed to: https://huggingface.co/{HF_REPO_ID}")


def main():
    parser = argparse.ArgumentParser(description="Push best model to HuggingFace")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint. Auto-selects best if not given.")
    args = parser.parse_args()
    push_best_model(args.ckpt)


if __name__ == "__main__":
    main()
