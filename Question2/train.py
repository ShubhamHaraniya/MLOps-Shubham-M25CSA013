import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import segmentation_models_pytorch as smp


# =====================================================
# CONFIG
# =====================================================
DATA_DIR = "."
RGB_DIR = os.path.join(DATA_DIR, "CameraRGB")
MASK_DIR = os.path.join(DATA_DIR, "CameraMask")

NUM_CLASSES = 23
IMAGE_SIZE = 256

EPOCHS = 15
BATCH_SIZE = 2          # safer for GPU memory
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# REPRODUCIBILITY
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# =====================================================
# DATASET
# =====================================================
class CityscapeDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ---------------- IMAGE ----------------
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # ---------------- MASK ----------------
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)

        # If mask has 3 channels, take red channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 2]

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert invalid labels to 255 (ignore)
        mask[(mask < 0) | (mask >= NUM_CLASSES)] = 255

        # ---------------- TO TENSOR ----------------
        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1) / 255.0

        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


# =====================================================
# METRICS
# =====================================================
def calculate_metrics(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    valid = (target >= 0) & (target < num_classes)

    pred = pred[valid]
    target = target[valid]

    if len(target) == 0:
        return 0.0, 0.0

    bins = num_classes * target + pred
    cm = torch.bincount(
        bins,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    present = (cm.sum(0) + cm.sum(1)) > 0

    miou = iou[present].mean().item() if present.any() else 0.0
    mdice = dice[present].mean().item() if present.any() else 0.0

    return miou, mdice


# =====================================================
# MAIN
# =====================================================
def main():
    print("Using Device:", DEVICE)

    # Check folders
    assert os.path.exists(RGB_DIR), f"{RGB_DIR} not found"
    assert os.path.exists(MASK_DIR), f"{MASK_DIR} not found"

    image_files = sorted([
        os.path.join(RGB_DIR, f)
        for f in os.listdir(RGB_DIR)
        if f.endswith(".png")
    ])

    mask_files = sorted([
        os.path.join(MASK_DIR, f)
        for f in os.listdir(MASK_DIR)
        if f.endswith(".png")
    ])

    print("Images :", len(image_files))
    print("Masks  :", len(mask_files))

    assert len(image_files) == len(mask_files), \
        "Image and mask count mismatch"

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        image_files,
        mask_files,
        test_size=0.2,
        random_state=42
    )

    # Dataset
    train_dataset = CityscapeDataset(X_train, y_train)
    test_dataset = CityscapeDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # =================================================
    # MODEL
    # =================================================
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # =================================================
    # TRAINING
    # =================================================
    losses = []
    mious = []
    mdices = []

    for epoch in range(EPOCHS):
        model.train()

        epoch_loss = 0
        epoch_miou = 0
        epoch_mdice = 0

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            miou, mdice = calculate_metrics(
                outputs.detach(),
                masks,
                NUM_CLASSES
            )

            epoch_loss += loss.item()
            epoch_miou += miou
            epoch_mdice += mdice

        avg_loss = epoch_loss / len(train_loader)
        avg_miou = epoch_miou / len(train_loader)
        avg_mdice = epoch_mdice / len(train_loader)

        losses.append(avg_loss)
        mious.append(avg_miou)
        mdices.append(avg_mdice)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"mIoU: {avg_miou:.4f} "
            f"mDice: {avg_mdice:.4f}"
        )

    # =================================================
    # SAVE MODEL
    # =================================================
    torch.save(model.state_dict(), "model.pth")
    print("Model saved: model.pth")

    # =================================================
    # PLOTS
    # =================================================
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")

    plt.figure(figsize=(10, 5))
    plt.plot(mious, label="mIoU")
    plt.plot(mdices, label="mDice")
    plt.title("Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("metrics_curve.png")

    # =================================================
    # TEST
    # =================================================
    model.eval()

    test_miou = 0
    test_mdice = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            miou, mdice = calculate_metrics(
                outputs,
                masks,
                NUM_CLASSES
            )

            test_miou += miou
            test_mdice += mdice

    test_miou /= len(test_loader)
    test_mdice /= len(test_loader)

    print("\nFinal Test Results")
    print("mIoU :", round(test_miou, 4))
    print("mDice:", round(test_mdice, 4))

    with open("metrics.txt", "w") as f:
        f.write(f"mIoU: {test_miou:.4f}\n")
        f.write(f"mDice: {test_mdice:.4f}\n")


if __name__ == "__main__":
    main()