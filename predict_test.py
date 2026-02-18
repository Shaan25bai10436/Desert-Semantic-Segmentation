import os
import cv2
import torch
import numpy as np

from model import SimpleSegmentationModel

# =====================
# CONFIG
# =====================
DEVICE = torch.device("cpu")
NUM_CLASSES = 6
MODEL_PATH = "segmentation_model_full.pth"

TEST_IMAGE_DIR = r"C:\Users\Lenovo\OneDrive\Desktop\Desert-Semantic-Segmentation\data\Offroad_Segmentation_Training_Dataset\test\Color_Images"
OUTPUT_DIR = "test_predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# LOAD MODEL
# =====================
model = SimpleSegmentationModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =====================
# INFERENCE
# =====================
for img_name in sorted(os.listdir(TEST_IMAGE_DIR)):
    img_path = os.path.join(TEST_IMAGE_DIR, img_name)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]

    image_resized = cv2.resize(image, (320, 180))
    image_resized = image_resized / 255.0

    tensor = torch.tensor(image_resized).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    save_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(save_path, pred.astype(np.uint8))

print("Test inference completed.")
print("Predicted masks saved in:", OUTPUT_DIR)

