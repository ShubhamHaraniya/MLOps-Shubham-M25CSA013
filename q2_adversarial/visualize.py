"""
Visualise adversarial examples from FGSM (scratch & ART), PGD, BIM.
Logs 10 samples per attack to WandB as required by the assignment.
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

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescentPyTorch,
    BasicIterativeMethod,
)

from config import (
    WANDB_PROJECT, WANDB_ENTITY, WEIGHT_DIR, RESULT_DIR,
    PGD_EPS, PGD_STEP, PGD_ITER,
    BIM_EPS, BIM_STEP, BIM_ITER,
)
from dataset import get_numpy_data, CIFAR10_CLASSES
from resnet_model import build_resnet18_cifar10
from fgsm_scratch import fgsm_attack_scratch


def load_art_classifier(device):
    ckpt_path = os.path.join(WEIGHT_DIR, "resnet18_cifar10_best.pt")
    model = build_resnet18_cifar10().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    art_cls = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return art_cls, model


def _to_display(x_np):
    """Convert CHW numpy (float32, 0-1) to HWC clipped uint8."""
    return (x_np.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)


def save_grid(clean, adv_dict, labels, title, path, n=10):
    """
    Save a visual comparison grid.
    clean    : (N, C, H, W) numpy
    adv_dict : {attack_name: (N, C, H, W) numpy}
    """
    n_attacks = len(adv_dict)
    n_rows    = 1 + n_attacks
    fig, axes = plt.subplots(n_rows, n, figsize=(n * 2, n_rows * 2 + 0.5))
    fig.suptitle(title, fontsize=12, y=1.01)

    for col in range(n):
        axes[0, col].imshow(_to_display(clean[col]))
        axes[0, col].axis("off")
        axes[0, col].set_title(CIFAR10_CLASSES[labels[col]], fontsize=7)
    axes[0, 0].set_ylabel("Original", fontsize=9)

    for row, (atk_name, adv) in enumerate(adv_dict.items(), start=1):
        for col in range(n):
            axes[row, col].imshow(_to_display(adv[col]))
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(atk_name, fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid: {path}")


def run(device):
    os.makedirs(RESULT_DIR, exist_ok=True)

    x_test, y_test = get_numpy_data(n_test=500)
    x_vis = x_test[:10]
    y_vis = y_test[:10]

    art_cls, model = load_art_classifier(device)
    criterion = nn.CrossEntropyLoss()

    # --- FGSM ART ---
    fgsm_art = FastGradientMethod(estimator=art_cls, eps=0.03)
    x_fgsm_art = fgsm_art.generate(x=x_vis)

    # --- FGSM Scratch ---
    imgs_t  = torch.tensor(x_vis).to(device)
    labs_t  = torch.tensor(y_vis).to(device)
    x_fgsm_scratch = fgsm_attack_scratch(
        model, imgs_t, labs_t, eps=0.03, criterion=criterion,
    ).cpu().numpy()

    # --- PGD ---
    pgd_attack = ProjectedGradientDescentPyTorch(
        estimator=art_cls, eps=PGD_EPS, eps_step=PGD_STEP,
        max_iter=PGD_ITER, targeted=False,
    )
    x_pgd = pgd_attack.generate(x=x_vis)

    # --- BIM ---
    bim_attack = BasicIterativeMethod(
        estimator=art_cls, eps=BIM_EPS, eps_step=BIM_STEP, max_iter=BIM_ITER,
    )
    x_bim = bim_attack.generate(x=x_vis)

    # --- save all grids ---
    adv_all = {
        "FGSM (scratch)": x_fgsm_scratch,
        "FGSM (ART)":     x_fgsm_art,
        "PGD":            x_pgd,
        "BIM":            x_bim,
    }
    grid_path = os.path.join(RESULT_DIR, "all_attacks_comparison.png")
    save_grid(x_vis, adv_all, y_vis, "Adversarial Attack Comparison", grid_path)

    # per-attack grids
    for atk_name, x_adv in adv_all.items():
        safe_name = atk_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        solo_path = os.path.join(RESULT_DIR, f"attack_{safe_name}.png")
        save_grid(x_vis, {atk_name: x_adv}, y_vis,
                  f"Original vs {atk_name}", solo_path)

    # --- perturbation heatmaps for one sample ---
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, (atk_name, x_adv) in zip(axes, adv_all.items()):
        diff = np.abs(x_adv[0] - x_vis[0]).mean(axis=0)
        im   = ax.imshow(diff, cmap="hot", vmin=0)
        ax.set_title(atk_name, fontsize=9); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.05)
    plt.suptitle("Perturbation Magnitude (avg over channels)", y=1.02)
    plt.tight_layout()
    heatmap_path = os.path.join(RESULT_DIR, "perturbation_heatmaps.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved heatmaps: {heatmap_path}")

    # --- WandB: log 10 samples per attack ---
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               name="adversarial_visualisation", reinit=True)

    for atk_name, x_adv in adv_all.items():
        images = []
        for i in range(10):
            c_img = _to_display(x_vis[i])
            a_img = _to_display(x_adv[i])
            lbl   = CIFAR10_CLASSES[y_vis[i]]
            images.append(wandb.Image(c_img, caption=f"clean/{lbl}"))
            images.append(wandb.Image(a_img, caption=f"{atk_name}/{lbl}"))
        safe = atk_name.replace(" ", "_").replace("(", "").replace(")", "")
        wandb.log({f"samples/{safe}": images})

    wandb.log({
        "comparison_grid":  wandb.Image(grid_path),
        "perturbation_heatmaps": wandb.Image(heatmap_path),
    })
    wandb.finish()

    print("\nVisualisation complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run(device)


if __name__ == "__main__":
    main()
