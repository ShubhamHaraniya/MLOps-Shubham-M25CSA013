"""
Train ResNet-34 adversarial detectors for PGD and BIM attacks.
Each detector is a binary classifier: clean=0, adversarial=1.
Target detection accuracy >= 70%.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    ProjectedGradientDescentPyTorch,
    BasicIterativeMethod,
)

from config import (
    WANDB_PROJECT, WANDB_ENTITY, WEIGHT_DIR, RESULT_DIR,
    PGD_EPS, PGD_STEP, PGD_ITER,
    BIM_EPS, BIM_STEP, BIM_ITER,
    DETECTOR_EPOCHS, DETECTOR_LR, BATCH_SIZE,
)
from dataset import get_numpy_data, get_numpy_train_data
from resnet_model import build_resnet18_cifar10, build_resnet34_detector


def load_resnet18_classifier(device):
    """Load trained ResNet-18 and wrap in ART classifier."""
    ckpt_path = os.path.join(WEIGHT_DIR, "resnet18_cifar10_best.pt")
    model = build_resnet18_cifar10().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return art_classifier


def generate_pgd(art_classifier, x_data):
    """Generate PGD adversarial examples."""
    print(f"Generating PGD adversarial examples (eps={PGD_EPS:.4f}, "
          f"step={PGD_STEP:.4f}, iter={PGD_ITER})...")
    attack = ProjectedGradientDescentPyTorch(
        estimator=art_classifier,
        eps=PGD_EPS,
        eps_step=PGD_STEP,
        max_iter=PGD_ITER,
        targeted=False,
    )
    return attack.generate(x=x_data)


def generate_bim(art_classifier, x_data):
    """Generate BIM adversarial examples."""
    print(f"Generating BIM adversarial examples (eps={BIM_EPS:.4f}, "
          f"step={BIM_STEP:.4f}, iter={BIM_ITER})...")
    attack = BasicIterativeMethod(
        estimator=art_classifier,
        eps=BIM_EPS,
        eps_step=BIM_STEP,
        max_iter=BIM_ITER,
    )
    return attack.generate(x=x_data)


def build_detector_dataset(x_clean, x_adv):
    """
    Combine clean (label=0) and adversarial (label=1) images.
    Returns balanced TensorDataset.
    """
    n = min(len(x_clean), len(x_adv))
    x = np.concatenate([x_clean[:n], x_adv[:n]], axis=0).astype(np.float32)
    y = np.array([0]*n + [1]*n, dtype=np.int64)

    # shuffle
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]

    return TensorDataset(torch.tensor(x), torch.tensor(y))


def train_detector(dataset, attack_name, device):
    """Train ResNet-34 binary detector and save weights."""
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    val_size   = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    model = build_resnet34_detector(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=DETECTOR_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=DETECTOR_EPOCHS)

    run_name = f"detector_{attack_name}"
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               name=run_name, reinit=True,
               config={"attack": attack_name, "epochs": DETECTOR_EPOCHS,
                       "lr": DETECTOR_LR, "train_size": train_size})

    best_val_acc = 0.0
    save_path = os.path.join(WEIGHT_DIR, f"detector_{attack_name}_best.pt")

    for epoch in range(1, DETECTOR_EPOCHS + 1):
        # train
        model.train()
        tr_correct, tr_total = 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"  [{attack_name}] epoch {epoch}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_total   += labels.size(0)

        # val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        tr_acc  = tr_correct  / tr_total
        val_acc = val_correct / val_total
        scheduler.step()

        print(f"  [{attack_name}] Epoch {epoch:02d}/{DETECTOR_EPOCHS}  "
              f"train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

        wandb.log({f"epoch": epoch,
                   f"train_acc": tr_acc, f"val_acc": val_acc,
                   f"lr": optimizer.param_groups[0]["lr"]})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(),
                        "val_acc": val_acc, "attack": attack_name,
                        "epoch": epoch}, save_path)
            print(f"    -> saved (val_acc={val_acc:.4f})")

    wandb.finish()
    print(f"[{attack_name}] Best val_acc = {best_val_acc:.4f}")

    if best_val_acc < 0.70:
        print(f"WARNING: [{attack_name}] detection accuracy below 70% target.")

    return save_path, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack",  choices=["pgd", "bim", "both"], default="both")
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--n_train", type=int, default=10000,
                        help="Number of training samples to use for adversarial generation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load ART-wrapped ResNet-18 for generating adversarial examples
    art_classifier = load_resnet18_classifier(device)

    # load data
    x_train, _     = get_numpy_train_data(n_train=args.n_train)
    x_test, y_test = get_numpy_data(n_test=2000)
    x_all = np.concatenate([x_train, x_test], axis=0)

    results = {}

    if args.attack in ("pgd", "both"):
        print("\n=== Training PGD Detector ===")
        x_pgd  = generate_pgd(art_classifier, x_all)
        dataset = build_detector_dataset(x_all, x_pgd)
        path, acc = train_detector(dataset, "pgd", device)
        results["pgd"] = acc
        # save generated adversarial examples for later evaluation
        np.save(os.path.join(WEIGHT_DIR, "x_pgd_adv.npy"), x_pgd[:2000])
        np.save(os.path.join(WEIGHT_DIR, "x_pgd_clean.npy"), x_all[:2000])

    if args.attack in ("bim", "both"):
        print("\n=== Training BIM Detector ===")
        x_bim  = generate_bim(art_classifier, x_all)
        dataset = build_detector_dataset(x_all, x_bim)
        path, acc = train_detector(dataset, "bim", device)
        results["bim"] = acc
        np.save(os.path.join(WEIGHT_DIR, "x_bim_adv.npy"), x_bim[:2000])
        np.save(os.path.join(WEIGHT_DIR, "x_bim_clean.npy"), x_all[:2000])

    print("\n=== Detector Training Summary ===")
    for attack, acc in results.items():
        status = "✓" if acc >= 0.70 else "✗"
        print(f"  {status} {attack.upper()} detector val_acc = {acc:.4f}")


if __name__ == "__main__":
    main()
