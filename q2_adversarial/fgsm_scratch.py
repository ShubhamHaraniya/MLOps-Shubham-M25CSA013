"""
FGSM attack implemented from scratch using PyTorch autograd.
x_adv = x + eps * sign(grad_x(Loss(f(x), y)))
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from config import (
    WANDB_PROJECT, WANDB_ENTITY, WEIGHT_DIR, RESULT_DIR,
    FGSM_EPSILONS, NUM_CLASSES_CIFAR10,
)
from dataset import get_dataloaders, CIFAR10_CLASSES
from resnet_model import build_resnet18_cifar10


def load_model(device):
    """Load the best trained ResNet-18."""
    ckpt_path = os.path.join(WEIGHT_DIR, "resnet18_cifar10_best.pt")
    model = build_resnet18_cifar10().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded ResNet-18 (test_acc={ckpt['test_acc']:.4f})")
    return model


def fgsm_attack_scratch(model, images, labels, eps, criterion):
    """
    FGSM from scratch.
    Inputs must be in [0,1] (unnormalized); model's NormalizeLayer handles it.
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad_sign = images.grad.data.sign()
    x_adv = images.detach() + eps * grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


@torch.no_grad()
def evaluate_clean_accuracy(model, loader, device):
    """Evaluate clean accuracy (no gradient needed)."""
    correct, total = 0, 0
    model.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        correct += (out.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def evaluate_adv_accuracy(model, loader, device, eps, criterion):
    """
    Evaluate accuracy under FGSM attack.
    Must NOT use torch.no_grad() because FGSM requires gradients.
    """
    correct, total = 0, 0
    model.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs_adv = fgsm_attack_scratch(model, imgs, labels, eps, criterion)
        with torch.no_grad():
            out = model(imgs_adv)
        correct += (out.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def run(device):
    os.makedirs(RESULT_DIR, exist_ok=True)

    _, test_loader = get_dataloaders(batch_size=64)
    model = load_model(device)
    criterion = nn.CrossEntropyLoss()

    # --- Clean accuracy ---
    clean_acc = evaluate_clean_accuracy(model, test_loader, device)
    print(f"Clean accuracy: {clean_acc:.4f}")

    # --- FGSM across epsilons ---
    results = []
    for eps in FGSM_EPSILONS:
        adv_acc = evaluate_adv_accuracy(model, test_loader, device, eps, criterion)
        drop = clean_acc - adv_acc
        results.append((eps, adv_acc, drop))
        print(f"  eps={eps:.3f}  adv_acc={adv_acc:.4f}  drop={drop:.4f}")

    # --- Save a batch of sample adversarial images ---
    sample_imgs, sample_labels = next(iter(test_loader))
    sample_imgs, sample_labels = sample_imgs.to(device), sample_labels.to(device)
    adv_samples_03 = fgsm_attack_scratch(model, sample_imgs, sample_labels,
                                         eps=0.03, criterion=criterion)

    _save_sample_comparison(
        sample_imgs[:10].cpu(), adv_samples_03[:10].cpu(),
        sample_labels[:10].cpu(), "fgsm_scratch",
        os.path.join(RESULT_DIR, "fgsm_scratch_samples.png"),
    )

    # --- Perturbation vs accuracy drop plot ---
    epsilons_, adv_accs, drops = zip(*results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epsilons_, adv_accs, marker="o", color="red")
    ax1.axhline(clean_acc, linestyle="--", color="gray", label="clean")
    ax1.set_xlabel("Epsilon"); ax1.set_ylabel("Accuracy"); ax1.legend()
    ax1.set_title("FGSM (scratch): Accuracy vs Epsilon")

    ax2.plot(epsilons_, drops, marker="o", color="orange")
    ax2.set_xlabel("Epsilon"); ax2.set_ylabel("Accuracy Drop")
    ax2.set_title("FGSM (scratch): Drop vs Epsilon")

    plt.tight_layout()
    plot_path = os.path.join(RESULT_DIR, "fgsm_scratch_eps_plot.png")
    fig.savefig(plot_path, dpi=150); plt.close(fig)
    print(f"Saved plot: {plot_path}")

    # --- WandB log ---
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               name="fgsm_scratch", reinit=True)
    wandb.log({
        "clean_acc": clean_acc,
        "fgsm_scratch/eps_plot": wandb.Image(plot_path),
        "fgsm_scratch/samples":  wandb.Image(
            os.path.join(RESULT_DIR, "fgsm_scratch_samples.png")),
    })
    for eps, adv_acc, drop in results:
        wandb.log({"epsilon": eps, "adv_acc_scratch": adv_acc, "drop": drop})
    wandb.finish()

    return clean_acc, results


def _save_sample_comparison(clean, adv, labels, title, path):
    """Save a grid: top=clean, bottom=adversarial."""
    n = len(clean)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        c_img = _to_displayable(clean[i])
        a_img = _to_displayable(adv[i])
        axes[0, i].imshow(c_img); axes[0, i].axis("off")
        axes[1, i].imshow(a_img); axes[1, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Clean",   fontsize=10)
            axes[1, i].set_ylabel(title,     fontsize=10)
        axes[0, i].set_title(CIFAR10_CLASSES[labels[i].item()], fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved comparison: {path}")


def _to_displayable(tensor):
    """Convert CHW tensor with values in [0,1] to HWC numpy."""
    return tensor.permute(1, 2, 0).numpy().clip(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run(device)


if __name__ == "__main__":
    main()
