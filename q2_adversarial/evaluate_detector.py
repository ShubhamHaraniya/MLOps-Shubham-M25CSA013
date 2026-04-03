"""
Evaluate adversarial detectors (PGD and BIM) on held-out test data.
Computes confusion matrix, precision, recall, detection accuracy.
Logs comparison to WandB.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score,
)
import wandb

from config import (
    WANDB_PROJECT, WANDB_ENTITY, WEIGHT_DIR, RESULT_DIR,
)
from resnet_model import build_resnet34_detector
from dataset import get_numpy_data, CIFAR10_CLASSES


def load_detector(attack_name, device):
    """Load a saved ResNet-34 detector checkpoint."""
    path = os.path.join(WEIGHT_DIR, f"detector_{attack_name}_best.pt")
    model = build_resnet34_detector(num_classes=2).to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {attack_name} detector (val_acc={ckpt['val_acc']:.4f})")
    return model


@torch.no_grad()
def predict_detector(model, x_data, device, batch_size=256):
    """Run detector inference, return predicted labels and probabilities."""
    dataset = TensorDataset(torch.tensor(x_data.astype(np.float32)))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_probs = [], []
    for (imgs,) in loader:
        imgs  = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs[:, 1].cpu().numpy())   # prob of being adversarial

    return np.concatenate(all_preds), np.concatenate(all_probs)


def evaluate_detector(attack_name, device):
    """Full evaluation pipeline for one attack."""
    os.makedirs(RESULT_DIR, exist_ok=True)

    # load adversarial + clean data saved during training
    x_adv_path   = os.path.join(WEIGHT_DIR, f"x_{attack_name}_adv.npy")
    x_clean_path = os.path.join(WEIGHT_DIR, f"x_{attack_name}_clean.npy")

    if not os.path.exists(x_adv_path):
        raise FileNotFoundError(
            f"{x_adv_path} not found. Run train_detector.py first.")

    x_adv   = np.load(x_adv_path).astype(np.float32)
    x_clean = np.load(x_clean_path).astype(np.float32)

    n = min(len(x_adv), len(x_clean))
    x_eval = np.concatenate([x_clean[:n], x_adv[:n]], axis=0)
    y_true = np.array([0]*n + [1]*n, dtype=np.int64)

    # shuffle
    idx = np.random.RandomState(0).permutation(len(x_eval))
    x_eval, y_true = x_eval[idx], y_true[idx]

    model = load_detector(attack_name, device)
    preds, probs = predict_detector(model, x_eval, device)

    test_acc = np.mean(preds == y_true)
    auc      = roc_auc_score(y_true, probs)
    report   = classification_report(
        y_true, preds, target_names=["clean", "adversarial"])

    print(f"\n[{attack_name.upper()}] Detection Accuracy : {test_acc:.4f}")
    print(f"[{attack_name.upper()}] ROC-AUC            : {auc:.4f}")
    print(report)

    # confusion matrix plot
    cm  = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["clean", "adversarial"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{attack_name.upper()} Detector — acc={test_acc:.3f}")
    plt.tight_layout()
    cm_path = os.path.join(RESULT_DIR, f"confusion_{attack_name}.png")
    fig.savefig(cm_path, dpi=150); plt.close(fig)
    print(f"Saved confusion matrix: {cm_path}")

    return {"attack": attack_name, "test_acc": test_acc,
            "auc": auc, "cm_path": cm_path, "report": report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", choices=["pgd", "bim", "both"], default="both")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    attacks = ["pgd", "bim"] if args.attack == "both" else [args.attack]

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               name="detector_evaluation", reinit=True)

    all_results = []
    for attack in attacks:
        try:
            res = evaluate_detector(attack, device)
            all_results.append(res)
            wandb.log({
                f"{attack}/test_acc":       res["test_acc"],
                f"{attack}/roc_auc":        res["auc"],
                f"{attack}/confusion_matrix": wandb.Image(res["cm_path"]),
            })
        except FileNotFoundError as e:
            print(f"Skipping {attack}: {e}")

    # comparison bar chart
    if len(all_results) > 1:
        names = [r["attack"].upper() for r in all_results]
        accs  = [r["test_acc"] for r in all_results]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(names, accs, color=["steelblue", "coral"])
        ax.set_ylim(0.5, 1.0)
        ax.axhline(0.70, linestyle="--", color="red", label="70% target")
        ax.set_ylabel("Detection Accuracy"); ax.set_title("Detector Comparison")
        ax.legend()
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=11)
        plt.tight_layout()
        comp_path = os.path.join(RESULT_DIR, "detector_comparison.png")
        fig.savefig(comp_path, dpi=150); plt.close(fig)
        wandb.log({"detector_comparison": wandb.Image(comp_path)})

    # summary table
    print("\n=== Detector Evaluation Summary ===")
    print(f"{'Attack':<8} {'Acc':>8} {'AUC':>8} {'>=70%'}")
    print("-" * 30)
    for r in all_results:
        ok = "✓" if r["test_acc"] >= 0.70 else "✗"
        print(f"{r['attack'].upper():<8} {r['test_acc']:>8.4f} {r['auc']:>8.4f} {ok}")

    wandb.finish()


if __name__ == "__main__":
    main()
