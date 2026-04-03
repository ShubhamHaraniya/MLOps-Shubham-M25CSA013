"""
Train ResNet-18 from scratch on CIFAR-10. Target: >= 72% test accuracy.
Logs to WandB. Saves best model weights.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import wandb

from config import (
    WANDB_PROJECT, WANDB_ENTITY, WEIGHT_DIR,
    TRAIN_EPOCHS, BATCH_SIZE, LR_RESNET, MOMENTUM, WEIGHT_DECAY,
)
from dataset import get_dataloaders
from resnet_model import build_resnet18_cifar10


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        name="resnet18_cifar10_clean_train",
        config={"epochs": args.epochs, "lr": LR_RESNET, "batch_size": BATCH_SIZE},
    )

    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = build_resnet18_cifar10().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LR_RESNET,
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # decay LR at epochs 15 and 25 (common schedule for 30-epoch CIFAR)
    scheduler = MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    best_acc = 0.0
    save_path = os.path.join(WEIGHT_DIR, "resnet18_cifar10_best.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = eval_epoch(model,  test_loader,  criterion,          device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"test_loss={te_loss:.4f}   test_acc={te_acc:.4f}")

        wandb.log({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr_loss, "train_acc": tr_acc,
            "test_loss":  te_loss, "test_acc":  te_acc,
        })

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"model_state_dict": model.state_dict(),
                        "test_acc": te_acc, "epoch": epoch}, save_path)
            print(f"  -> saved best  (test_acc={te_acc:.4f})")

    wandb.finish()
    print(f"\nBest test accuracy: {best_acc:.4f}")
    if best_acc < 0.72:
        print("WARNING: did not reach 72% target — try more epochs or tune LR.")


if __name__ == "__main__":
    main()
