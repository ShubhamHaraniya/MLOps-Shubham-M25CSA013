"""
Test script for ViT-S on CIFAR-100.
Computes overall + class-wise accuracy and generates histogram.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from collections import defaultdict

from config import (
    get_experiment_config, get_run_name, LORA_EXPERIMENTS,
    WANDB_PROJECT, WANDB_ENTITY, WEIGHT_DIR, RESULT_DIR, NUM_CLASSES,
)
from dataset import get_dataloaders
from model import create_model, print_trainable_params
from torchvision.datasets import CIFAR100


def load_best_checkpoint(exp_id):
    """Find and load the best checkpoint for a given experiment."""
    pattern = os.path.join(WEIGHT_DIR, f"exp{exp_id}_*_best.pt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for exp {exp_id} at {pattern}")
    ckpt_path = matches[0]
    print(f"Loading checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


@torch.no_grad()
def test_model(model, test_loader, device):
    """Evaluate model on test set, return overall acc and per-class acc."""
    model.eval()
    correct, total = 0, 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for pred, label in zip(preds, labels):
            class_total[label.item()] += 1
            if pred == label:
                class_correct[label.item()] += 1

    overall_acc = correct / total
    per_class_acc = {
        c: class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        for c in range(NUM_CLASSES)
    }
    return overall_acc, per_class_acc


def plot_classwise_histogram(per_class_acc, exp_id, run_name, save_dir):
    """Generate and save class-wise accuracy histogram."""
    os.makedirs(save_dir, exist_ok=True)

    classes = sorted(per_class_acc.keys())
    accs = [per_class_acc[c] for c in classes]

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.barplot(x=list(range(len(classes))), y=accs, ax=ax, color="steelblue")
    ax.set_xlabel("Class Index")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Class-wise Test Accuracy — Exp {exp_id}: {run_name}")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=np.mean(accs), color="red", linestyle="--", label=f"Mean={np.mean(accs):.3f}")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, f"classwise_acc_exp{exp_id}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram to {path}")
    return path


def run_test(exp_id, device="cuda"):
    """Test a single experiment and log results."""
    cfg = get_experiment_config(exp_id)
    run_name = get_run_name(cfg)
    print(f"\nTesting Exp {exp_id}: {run_name}")

    # load checkpoint
    ckpt = load_best_checkpoint(exp_id)

    # rebuild model
    model = create_model(
        num_classes=cfg["num_classes"],
        use_lora=cfg["use_lora"],
        rank=cfg["rank"],
        alpha=cfg["alpha"],
        dropout=cfg["dropout"],
        model_name=cfg["model_name"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    trainable, total = print_trainable_params(model)
    model = model.to(device)

    # data
    _, _, test_loader = get_dataloaders(batch_size=cfg["batch_size"])

    # test
    overall_acc, per_class_acc = test_model(model, test_loader, device)
    print(f"  Overall Test Accuracy: {overall_acc:.4f}")

    # histogram
    os.makedirs(RESULT_DIR, exist_ok=True)
    hist_path = plot_classwise_histogram(per_class_acc, exp_id, run_name, RESULT_DIR)

    # wandb logging
    wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        name=f"test_{run_name}", config=cfg, reinit=True,
    )
    wandb.log({
        "test_accuracy": overall_acc,
        "trainable_params": trainable,
        "classwise_histogram": wandb.Image(hist_path),
    })
    wandb.finish()

    return {
        "exp_id": exp_id,
        "use_lora": cfg["use_lora"],
        "rank": cfg["rank"],
        "alpha": cfg["alpha"],
        "dropout": cfg["dropout"],
        "test_accuracy": overall_acc,
        "trainable_params": trainable,
    }


def print_results_table(results):
    """Print markdown-style results table."""
    header = "| LoRA | Rank | Alpha | Dropout | Test Accuracy | Trainable Params |"
    sep = "|------|------|-------|---------|--------------|-----------------|"
    print(f"\n{header}\n{sep}")
    for r in results:
        lora_str = "Yes" if r["use_lora"] else "No"
        rank_str = str(r["rank"]) if r["rank"] else "—"
        alpha_str = str(r["alpha"]) if r["alpha"] else "—"
        drop_str = str(r["dropout"]) if r["dropout"] else "—"
        print(f"| {lora_str:4} | {rank_str:4} | {alpha_str:5} | {drop_str:7} "
              f"| {r['test_accuracy']:.4f}       | {r['trainable_params']:>15,} |")


def main():
    parser = argparse.ArgumentParser(description="Test ViT-S on CIFAR-100")
    parser.add_argument("--exp", type=int, default=-1,
                        help="Experiment id (0-9). -1 = test all.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.exp == -1:
        results = []
        for eid, *_ in LORA_EXPERIMENTS:
            try:
                res = run_test(eid, device=device)
                results.append(res)
            except FileNotFoundError as e:
                print(f"  Skipping exp {eid}: {e}")
        print_results_table(results)
    else:
        res = run_test(args.exp, device=device)
        print_results_table([res])


if __name__ == "__main__":
    main()
