import os
import cv2
import torch
from torch.utils.data import Dataset


class FalconDesertDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

        # Map sparse class IDs to continuous IDs
        self.class_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            27: 4,
            39: 5
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize for faster CPU training
        image = cv2.resize(image, (320, 180))
        mask = cv2.resize(mask, (320, 180), interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image / 255.0

        # Convert to tensors
        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()

        # Apply class mapping
        mapped_mask = torch.zeros_like(mask)
        for original_id, new_id in self.class_map.items():
            mapped_mask[mask == original_id] = new_id

        return image, mapped_mask
