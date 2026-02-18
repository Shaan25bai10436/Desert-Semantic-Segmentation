import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import SimpleSegmentationModel

# =====================
# CONFIG
# =====================
DEVICE = torch.device("cpu")
NUM_CLASSES = 6
MODEL_PATH = "segmentation_model_full.pth"

TEST_IMAGE_DIR = r"C:\Users\Lenovo\OneDrive\Desktop\desert_segmentation\data\Offroad_Segmentation_Training_Dataset\test\Color_Images"

# =====================
# FIND ONE TEST IMAGE
# =====================
image_files = sorted([
    f for f in os.listdir(TEST_IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if len(image_files) == 0:
    raise RuntimeError("No images found in test Color_Images folder")

IMAGE_PATH = os.path.join(TEST_IMAGE_DIR, image_files[0])

print("Visualizing image:", IMAGE_PATH)

# =====================
# LOAD MODEL
# =====================
model = SimpleSegmentationModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =====================
# LOAD IMAGE
# =====================
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w = image.shape[:2]

image_resized = cv2.resize(image, (320, 180))
image_resized = image_resized / 255.0

tensor = torch.tensor(image_resized).permute(2, 0, 1).float()
tensor = tensor.unsqueeze(0).to(DEVICE)

# =====================
# PREDICT
# =====================
with torch.no_grad():
    output = model(tensor)
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

# =====================
# COLOR MAP
# =====================
colors = np.array([
    [0, 0, 0],       # class 0
    [255, 0, 0],     # class 1
    [0, 255, 0],     # class 2
    [0, 0, 255],     # class 3
    [255, 255, 0],   # class 4
    [0, 255, 255],   # class 5
])

color_mask = colors[pred]
overlay = (0.6 * image + 0.4 * color_mask).astype(np.uint8)

# =====================
# DISPLAY
# =====================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predicted Segmentation")
plt.imshow(color_mask)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()
