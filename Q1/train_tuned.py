import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Configuration - TUNED
# ==========================
TRAIN_DIR = "data/data/train/"
TEST_DIR = "data/data/test/"
BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset & DataLoader
# ==========================
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print("Classes:", class_names)

# ==========================
# Load ResNet-18
# ==========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ==========================
# Loss & Optimizer (SGD with momentum for better convergence)
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

# ==========================
# Training Loop
# ==========================
train_losses = []
train_accs = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {epoch_loss:.4f} "
          f"Accuracy: {epoch_acc:.2f}%")

print("Training Complete!")

# ==========================
# Save Model
# ==========================
torch.save(model.state_dict(), "trained_model_tuned.pth")
print("Tuned model saved!")

# ==========================
# Plot Training Curves
# ==========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(1, EPOCHS+1), train_losses, 'b-o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True)

ax2.plot(range(1, EPOCHS+1), train_accs, 'r-o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy')
ax2.grid(True)

plt.suptitle(f'Training Curves (LR={LR}, BS={BATCH_SIZE}, Optimizer=SGD)')
plt.tight_layout()
plt.savefig('training_curves_tuned.png', dpi=150)
print("Training curves saved to training_curves_tuned.png")

# ==========================
# Evaluate on Test Set
# ==========================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

overall_acc = accuracy_score(all_labels, all_preds)
print(f"\nOverall Test Accuracy: {overall_acc * 100:.2f}%")

macro_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"F1 Score: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ==========================
# Classwise Accuracy
# ==========================
print("\n=== Classwise Accuracy ===")
cm = confusion_matrix(all_labels, all_preds)
for i, cls_name in enumerate(class_names):
    cls_total = cm[i].sum()
    cls_correct = cm[i][i]
    cls_acc = cls_correct / cls_total * 100 if cls_total > 0 else 0
    print(f"Class {cls_name} Accuracy: {cls_acc:.2f}%")

# ==========================
# Confusion Matrix Plot
# ==========================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Tuned Model)')
plt.tight_layout()
plt.savefig('confusion_matrix_tuned.png', dpi=150)
print("\nConfusion matrix saved to confusion_matrix_tuned.png")
