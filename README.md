# Offroad Autonomy: Desert Semantic Segmentation# Synopsis
A **baseline semantic segmentation model** is implemented in this project for an offroad

**Duality AI (Falcon platform)** supplied the dataset for the desert environment.

To enable fine-grained scene understanding for offroad autonomy, the objective is to assign **each pixel** in an image to one of **10 terrain/object classes**.

The project's main objectives are:
Appropriate data management
A hygienic and repeatable training pipeline
A straightforward, reliable model that performs well in generalizing to unobserved desert landscapes

--- ## Problem Description
Offroad autonomous systems need to comprehend complicated terrain at the pixel level in order to navigate safely and avoid obstacles.

Considering a fictitious desert dataset:
- Use labeled images to train a semantic segmentation model.
Assess performance using unobserved photos from a comparable desert setting.
Aim for the **Intersection over Union (IoU)** score.
--- ##Dataset
The Duality AI (Falcon) synthetic off-road desert dataset is the source.
The dataset is utilized **exactly as supplied by the organizers**.
No changes are made to the dataset.## Dataset Structure data/├── train/│ ├── images/│ └── masks/
├── images/│ └── masks/ ├── val/
Test/
└── pictures/### Properties of the Dataset
- Pictures: RGB
- Masks: Grayscale single-channel
A **class ID** is associated with each pixel value in the mask.
The filenames of the image and mask match exactly.
**No masks** are present in the test set; they are only utilized for inference.

--- ##Classes
One of the following **10 classes per pixel** is predicted by the model: Class ID | Class Name | |--------|-----------| 0 | Trees || 1 | Lush Bushes || 2 | Dry Grass || 3 | Dry Bushes || 4 | Ground Clutter || 5 | Flowers || 6 | Logs || 7 | Rocks || 8 | Landscape || 9 | Sky |
--- ## Project Structure desert_segmentation/ ├── data/ ├── dataset.py ├── model.py ├── train.py ├── utils.py └── README.md ### File Description - `dataset.py` – Dataset loading and preprocessing logic - `model.py` – Simple encoder–decoder segmentation model - `train.py` – Training and validation loop - `utils.py` – IoU computation and helper functions - `data/` – Dataset directory (train / val / test) --- ## Model - Architecture: **Simple encoder–decoder segmentation model (U-Net–style baseline)** - Framework: **PyTorch** - Designed to be: - Lightweight - Stable - Easy to understand - Suitable for beginners The model outputs **raw class logits** for each pixel. --- ## Training Details - Loss Function: `CrossEntropyLoss` - Optimizer: Adam - Input: RGB image tensor `[3, H, W]` - Output: Class score tensor `[10, H, W]` - Masks are used as integer class labels (not normalized or recolored) --- ## Evaluation - Primary metric: **Intersection over Union (IoU)** - IoU is computed per class and averaged - Focus is on generalization to unseen desert environment.

--- ## The ability to reproduce
1. Insert the dataset into the `data/` directory.
2. Verify that the folder structure adheres to the desired arrangement.
3. Execute the training script: ```bash python train.py
