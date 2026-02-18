# Desert-Semantic-Segmentation
This project implements a **baseline semantic segmentation model** for an offroad desert environment dataset provided by **Duality AI (Falcon platform)**.  The goal is to classify **each pixel** in an image into one of **10 terrain/object classes**, enabling fine-grained scene understanding for offroad autonomy.
The goal is to classify **each pixel** in an image into one of **10 terrain/object
classes**, enabling fine-grained scene understanding for offroad autonomy.

The project focuses on:
- Correct data handling
- A clean and reproducible training pipeline
- A simple, stable model that generalizes well to unseen desert scenes

---

## Problem Statement
Offroad autonomous systems require pixel-level understanding of complex terrain
to enable safe navigation and obstacle avoidance.

Given a synthetic desert dataset:
- Train a semantic segmentation model on labeled images
- Evaluate performance on unseen images from a similar desert environment
- Optimize for **Intersection over Union (IoU)** score

---

## Dataset
- Source: Duality AI (Falcon) synthetic offroad desert dataset
- Dataset is used **exactly as provided by the organizers**
- No dataset modification is performed

### Dataset Structure
data/
├── train/
│ ├── images/
│ └── masks/
├── val/
│ ├── images/
│ └── masks/
└── test/
└── images/
### Dataset Properties
- Images: RGB
- Masks: Single-channel grayscale
- Each pixel value in the mask corresponds to a **class ID**
- Image and mask filenames match exactly
- Test set contains **no masks** (used only for inference)

---

## Classes
The model predicts one of the following **10 classes per pixel**:

| Class ID | Class Name |
|--------|-----------|
| 0 | Trees |
| 1 | Lush Bushes |
| 2 | Dry Grass |
| 3 | Dry Bushes |
| 4 | Ground Clutter |
| 5 | Flowers |
| 6 | Logs |
| 7 | Rocks |
| 8 | Landscape |
| 9 | Sky |

---

## Project Structure
desert_segmentation/
├── data/
├── dataset.py
├── model.py
├── train.py
├── utils.py
└── README.md
### File Description
- `dataset.py` – Dataset loading and preprocessing logic
- `model.py` – Simple encoder–decoder segmentation model
- `train.py` – Training and validation loop
- `utils.py` – IoU computation and helper functions
- `data/` – Dataset directory (train / val / test)

---

## Model
- Architecture: **Simple encoder–decoder segmentation model (U-Net–style baseline)**
- Framework: **PyTorch**
- Designed to be:
  - Lightweight
  - Stable
  - Easy to understand
  - Suitable for beginners

The model outputs **raw class logits** for each pixel.

---

## Training Details
- Loss Function: `CrossEntropyLoss`
- Optimizer: Adam
- Input: RGB image tensor `[3, H, W]`
- Output: Class score tensor `[10, H, W]`
- Masks are used as integer class labels (not normalized or recolored)

---

## Evaluation
- Primary metric: **Intersection over Union (IoU)**
- IoU is computed per class and averaged
- Focus is on generalization to unseen desert environments

---

## Reproducibility
1. Place the dataset inside the `data/` directory
2. Ensure folder structure matches the expected layout
3. Run the training script:
   ```bash
   python train.py
   
