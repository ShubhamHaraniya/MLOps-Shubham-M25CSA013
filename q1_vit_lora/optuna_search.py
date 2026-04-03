"""
Optuna hyperparameter search for LoRA on CIFAR-100 ViT-S.
Searches over rank, alpha, dropout, and learning rate.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import optuna
import wandb

from config import (
    WANDB_PROJECT, WANDB_ENTITY, WEIGHT_DIR, NUM_CLASSES,
    EPOCHS, BATCH_SIZE, MODEL_NAME,
)
from dataset import get_dataloaders
from model import create_model, print_trainable_params

OPTUNA_EPOCHS = 5       # quick eval per trial
OPTUNA_TRIALS = 20
OPTUNA_STUDY  = "lora_hpo"


def objective(trial, device):
    """Optuna objective: returns best val accuracy."""
    rank    = trial.suggest_categorical("rank",    [2, 4, 8, 16, 32])
    alpha   = trial.suggest_categorical("alpha",   [2, 4, 8, 16, 32])
    dropout = trial.suggest_float("dropout",        0.0, 0.3, step=0.05)
    lr      = trial.suggest_float("lr",             1e-5, 5e-4, log=True)

    run_name = f"optuna_r{rank}_a{alpha}_d{dropout:.2f}_lr{lr:.0e}"

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={"rank": rank, "alpha": alpha, "dropout": dropout, "lr": lr,
                "optuna_trial": trial.number},
        reinit=True,
    )

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    model = create_model(
        num_classes=NUM_CLASSES,
        use_lora=True,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        model_name=MODEL_NAME,
    )
    model = model.to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=OPTUNA_EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, OPTUNA_EPOCHS + 1):
        # --- train ---
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # --- validate ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total
        scheduler.step()

        wandb.log({"epoch": epoch, "val_acc": val_acc})

        # optuna pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        best_val_acc = max(best_val_acc, val_acc)

    wandb.finish()
    return best_val_acc


def run_full_experiment(best_params, device):
    """Re-train best config for full EPOCHS and save model."""
    from train import run_experiment
    import config as cfg_module

    # Temporarily override config so train.py can see a new "best" experiment
    print(f"\nRe-training best config for {EPOCHS} epochs: {best_params}")
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    model = create_model(
        num_classes=NUM_CLASSES,
        use_lora=True,
        rank=best_params["rank"],
        alpha=best_params["alpha"],
        dropout=best_params["dropout"],
        model_name=MODEL_NAME,
    )
    print_trainable_params(model)
    model = model.to(device)

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=best_params["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        name="optuna_best_full_run", config=best_params, reinit=True,
    )

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total
        scheduler.step()

        wandb.log({"epoch": epoch, "val_acc": val_acc})
        print(f"  Epoch {epoch}/{EPOCHS}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(WEIGHT_DIR, "optuna_best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "config": best_params,
            }, save_path)
            print(f"  -> saved to {save_path}")

    wandb.finish()
    print(f"Best val_acc = {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=OPTUNA_TRIALS)
    parser.add_argument("--device",  type=str, default="cuda")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain best found config for full epochs after search")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    study = optuna.create_study(
        study_name=OPTUNA_STUDY,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(lambda t: objective(t, device), n_trials=args.trials)

    print("\n=== Optuna Best Trial ===")
    best = study.best_trial
    print(f"  Val accuracy : {best.value:.4f}")
    print(f"  Params       : {best.params}")

    if args.retrain:
        run_full_experiment(best.params, device)


if __name__ == "__main__":
    main()
