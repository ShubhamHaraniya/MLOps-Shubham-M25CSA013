"""
FGSM attack using IBM Adversarial Robustness Toolbox (ART).
Compares: clean vs FGSM-scratch vs FGSM-ART accuracy.
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
from art.attacks.evasion import FastGradientMethod

from config import (
    WANDB_PROJECT, WANDB_ENTITY, WEIGHT_DIR, RESULT_DIR, FGSM_EPSILONS,
)
from dataset import get_numpy_data, CIFAR10_CLASSES
from resnet_model import build_resnet18_cifar10
from fgsm_scratch import run as run_scratch, _save_sample_comparison, _to_displayable


def load_art_classifier(device):
    """Wrap ResNet-18 in ART PyTorchClassifier."""
    ckpt_path = os.path.join(WEIGHT_DIR, "resnet18_cifar10_best.pt")
    model = build_resnet18_cifar10().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    print(f"ART classifier loaded (test_acc={ckpt['test_acc']:.4f})")
    return classifier, model


def art_fgsm_accuracy(classifier, x_test, y_test, eps):
    """Generate ART FGSM adversarial examples and measure accuracy."""
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    x_adv  = attack.generate(x=x_test)

    preds    = np.argmax(classifier.predict(x_adv), axis=1)
    accuracy = np.mean(preds == y_test)
    return x_adv, accuracy


def run(device):
    os.makedirs(RESULT_DIR, exist_ok=True)

    # load data as numpy (ART friendly, [0,1])
    x_test, y_test = get_numpy_data(n_test=2000)
    print(f"Test data: {x_test.shape}, {y_test.shape}")

    classifier, model = load_art_classifier(device)

    # clean accuracy via ART
    preds_clean = np.argmax(classifier.predict(x_test), axis=1)
    clean_acc   = np.mean(preds_clean == y_test)
    print(f"Clean accuracy (ART): {clean_acc:.4f}")

    # sweep epsilons
    art_results = []
    for eps in FGSM_EPSILONS:
        x_adv, adv_acc = art_fgsm_accuracy(classifier, x_test, y_test, eps)
        drop = clean_acc - adv_acc
        art_results.append((eps, adv_acc, drop))
        print(f"  eps={eps:.3f}  adv_acc={adv_acc:.4f}  drop={drop:.4f}")

    # save 10 sample comparisons (eps=0.03)
    attack_vis = FastGradientMethod(estimator=classifier, eps=0.03)
    x_adv_vis  = attack_vis.generate(x=x_test[:10])

    clean_t = torch.tensor(x_test[:10])
    adv_t   = torch.tensor(x_adv_vis)
    labels_t = torch.tensor(y_test[:10])

    _save_sample_comparison(
        clean_t, adv_t, labels_t, "FGSM-ART",
        os.path.join(RESULT_DIR, "fgsm_art_samples.png"),
    )

    # overlay comparison: scratch vs ART
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    eps_vals    = [e for e, _, _ in art_results]
    art_accs    = [a for _, a, _ in art_results]
    axes.plot(eps_vals, art_accs, marker="s", color="blue",  label="FGSM-ART")
    axes.axhline(clean_acc, linestyle="--", color="gray", label="clean")
    axes.set_xlabel("Epsilon"); axes.set_ylabel("Accuracy")
    axes.set_title("FGSM‑ART: Accuracy vs Epsilon")
    axes.legend(); plt.tight_layout()
    plot_path = os.path.join(RESULT_DIR, "fgsm_art_eps_plot.png")
    fig.savefig(plot_path, dpi=150); plt.close(fig)

    # combined comparison plot (requires scratch results file if available)
    _plot_combined(clean_acc, art_results)

    # WandB
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               name="fgsm_art_comparison", reinit=True)
    wandb.log({
        "clean_acc": clean_acc,
        "fgsm_art/eps_plot":  wandb.Image(plot_path),
        "fgsm_art/samples":   wandb.Image(
            os.path.join(RESULT_DIR, "fgsm_art_samples.png")),
    })
    for eps, adv_acc, drop in art_results:
        wandb.log({"epsilon": eps, "adv_acc_art": adv_acc, "drop_art": drop})

    # --- Log 10 FGSM samples to WandB (with/without ART) ---
    _log_wandb_samples(x_test, x_adv_vis, y_test)

    wandb.finish()
    return clean_acc, art_results


def _plot_combined(clean_acc, art_results):
    """Plot clean, ART adversarial on one figure."""
    eps_vals = [e for e, _, _ in art_results]
    art_accs = [a for _, a, _ in art_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_vals, art_accs, marker="s", color="blue",  label="FGSM-ART")
    ax.axhline(clean_acc, linestyle="--", color="gray", label="clean")
    ax.set_xlabel("Epsilon"); ax.set_ylabel("Accuracy")
    ax.set_title("FGSM Attack Comparison: Clean vs ART")
    ax.legend(); plt.tight_layout()
    path = os.path.join(RESULT_DIR, "fgsm_combined_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved combined plot: {path}")


def _log_wandb_samples(x_clean, x_adv, y_labels, n=10):
    """Log n side-by-side image pairs to WandB."""
    images = []
    for i in range(min(n, len(x_clean))):
        c_img = (x_clean[i].transpose(1, 2, 0) * 255).astype(np.uint8)
        a_img = (x_adv[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        images.append(wandb.Image(c_img, caption=f"clean/{CIFAR10_CLASSES[y_labels[i]]}"))
        images.append(wandb.Image(a_img, caption=f"FGSM-ART/{CIFAR10_CLASSES[y_labels[i]]}"))
    wandb.log({"fgsm_art_samples_wandb": images})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run(device)


if __name__ == "__main__":
    main()
