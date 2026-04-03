"""
ViT-S model creation with optional LoRA injection via PEFT.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model


def create_model(num_classes=100, use_lora=False, rank=None, alpha=None,
                 dropout=None, model_name="vit_small_patch16_224"):
    """
    Load pretrained ViT-S and optionally inject LoRA adapters.

    Without LoRA: freeze backbone, only classification head is trainable.
    With LoRA: freeze backbone, inject LoRA into attn.qkv layers,
               classification head stays trainable via modules_to_save.
    """
    # load imagenet-pretrained vit-small
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    if not use_lora:
        # freeze everything except the head
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        return model

    # with LoRA — freeze backbone, peft handles the rest
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["qkv"],   # timm ViT uses fused qkv projection
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["head"],  # keep classification head trainable
    )

    model = get_peft_model(model, lora_config)
    return model


def print_trainable_params(model):
    """Print number of trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total
    print(f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
    return trainable, total
