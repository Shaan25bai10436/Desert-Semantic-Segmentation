import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FalconDesertDataset
from model import SimpleSegmentationModel
from utils import compute_average_loss, compute_iou

# =====================
# CONFIG
# =====================
DEVICE = torch.device("cpu")
NUM_CLASSES = 6
EPOCHS = 20
MODEL_PATH = "segmentation_model_full.pth"

# =====================
# DATASETS
# =====================
train_dataset = FalconDesertDataset(
    "data/Offroad_Segmentation_Training_Dataset/train/Color_Images",
    "data/Offroad_Segmentation_Training_Dataset/train/Segmentation"
)

val_dataset = FalconDesertDataset(
    "data/Offroad_Segmentation_Training_Dataset/val/Color_Images",
    "data/Offroad_Segmentation_Training_Dataset/val/Segmentation"
)

# =====================
# LOADERS
# =====================
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

# =====================
# MODEL
# =====================
model = SimpleSegmentationModel(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =====================
# TRAINING
# =====================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_loss = compute_average_loss(model, val_loader, criterion, DEVICE)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val   Loss: {val_loss:.4f}")

# =====================
# SAVE MODEL
# =====================
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# =====================
# FINAL IoU
# =====================
ious = compute_iou(model, val_loader, NUM_CLASSES, DEVICE)

print("\nIoU per class:")
for i, v in enumerate(ious):
    print(f"Class {i}: {v:.4f}")

print(f"\nMean IoU: {ious.mean():.4f}")
