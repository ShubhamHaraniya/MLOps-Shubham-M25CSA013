"""
Training script for ViT-S on CIFAR-100 with/without LoRA.
Logs to WandB: loss, accuracy, gradient norms on LoRA weights.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

# Explicitly authenticate — required inside SLURM (no interactive shell)
_WANDB_KEY = os.environ.get("WANDB_API_KEY")
if _WANDB_KEY:
    wandb.login(key=_WANDB_KEY, relogin=True)

from config import (
    get_experiment_config, get_run_name, LORA_EXPERIMENTS,
    WANDB_PROJECT, WANDB_ENTITY, WEIGHT_DIR, RESULT_DIR,
)
from dataset import get_dataloaders
from model import create_model, print_trainable_params


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch, return avg loss and accuracy."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model, return avg loss and accuracy."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def get_lora_grad_norms(model):
    """Collect gradient L2 norms for LoRA parameters."""
    norms = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.grad is not None:
            norms[f"grad_norm/{name}"] = param.grad.norm(2).item()
    return norms


def run_experiment(exp_id, device="cuda"):
    """Train a single experiment configuration."""
    cfg = get_experiment_config(exp_id)
    run_name = get_run_name(cfg)
    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}: {run_name}")
    print(f"{'='*60}")

    # init wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=cfg,
        reinit="finish_previous",
    )

    # data
    train_loader, val_loader, _ = get_dataloaders(batch_size=cfg["batch_size"])

    # model
    model = create_model(
        num_classes=cfg["num_classes"],
        use_lora=cfg["use_lora"],
        rank=cfg["rank"],
        alpha=cfg["alpha"],
        dropout=cfg["dropout"],
        model_name=cfg["model_name"],
    )
    trainable, total = print_trainable_params(model)
    wandb.config.update({"trainable_params": trainable, "total_params": total})
    model = model.to(device)

    # training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_val_acc = 0.0
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )

        # collect gradient norms before optimizer.zero_grad in next iter
        grad_norms = get_lora_grad_norms(model) if cfg["use_lora"] else {}

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # log to wandb
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_dict.update(grad_norms)
        wandb.log(log_dict)

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
        print(f"  val_loss={val_loss:.4f}    val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(WEIGHT_DIR, f"exp{exp_id}_{run_name}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "config": cfg,
            }, ckpt_path)
            print(f"  -> saved best model (val_acc={val_acc:.4f})")

    wandb.finish()
    print(f"Experiment {exp_id} done. Best val_acc = {best_val_acc:.4f}")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train ViT-S on CIFAR-100")
    parser.add_argument("--exp", type=int, default=-1,
                        help="Experiment id (0-9). -1 = run all.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.exp == -1:
        # run all experiments
        results = {}
        for eid, *_ in LORA_EXPERIMENTS:
            acc = run_experiment(eid, device=device)
            results[eid] = acc
        print("\n\nAll Results:")
        for eid, acc in results.items():
            cfg = get_experiment_config(eid)
            print(f"  Exp {eid} ({get_run_name(cfg)}): val_acc={acc:.4f}")
    else:
        run_experiment(args.exp, device=device)


if __name__ == "__main__":
    main()
