# Desert / Offroad Semantic Segmentation

This project implements a **semantic segmentation** pipeline for offroad/desert scenes using **PyTorch**. The goal is to assign a class label to **every pixel** in an image. The project includes training, validation, test-time inference, and visualization.

---

## 1. Problem Statement

Given RGB images of offroad/desert environments, predict a **pixel-wise segmentation mask** where each pixel belongs to one of the predefined classes. The model is evaluated using **Intersection over Union (IoU)** on a validation set. The test set is used **only for inference**.

---

## 2. Dataset Structure

The dataset is organized as follows:

```
data/
└── Offroad_Segmentation_Training_Dataset/
    ├── train/
    │   ├── Color_Images/
    │   └── Segmentation/
    ├── val/
    │   ├── Color_Images/
    │   └── Segmentation/
    └── test/
        ├── Color_Images/
        └── Segmentation/   (not used)
```

**Important rules followed:**
- Training and validation masks are used only during training/validation.
- Test masks (if present) are **not used**.
- Test images are used **only for inference**.

---

## 3. Classes

Original dataset labels were sparse. They were mapped to **contiguous class IDs** for training:

| New ID | Description |
|------:|-------------|
| 0 | Class 0 |
| 1 | Class 1 |
| 2 | Class 2 |
| 3 | Class 3 |
| 4 | Class 27 (mapped) |
| 5 | Class 39 (mapped) |

---

## 4. Project Structure

```
desert_segmentation/
├── dataset.py                 # Dataset loader and label mapping
├── model.py                   # Segmentation model
├── utils.py                   # Metrics (IoU)
├── train.py                   # Training and validation
├── predict_test.py            # Test-time inference
├── visualise.py               # Visualization for demo
├── segmentation_model_full.pth# Final trained model
├── test_predictions/          # Predicted test masks
├── data/
└── README.md
```

---

## 5. Model

- Framework: **PyTorch**
- Architecture: **Simple encoder–decoder segmentation network**
- Input size (during training): resized for efficiency
- Output: `num_classes` feature maps (one per class)

The model predicts a class label for each pixel.

---

## 6. Training Details

- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Device: CPU
- Epochs: 20
- Input normalization: images scaled to `[0, 1]`
- Masks: integer class IDs (not normalized)

Training and validation are strictly separated.

---

## 7. Evaluation Metric

**Accuracy is not used** for semantic segmentation.

The project is evaluated using:

- **Intersection over Union (IoU)** per class
- **Mean IoU (mIoU)** across classes

### Final Validation Performance

```
Mean IoU ≈ 0.37
```

This represents a **baseline-level performance** using a simple model without heavy optimization.

---

## 8. Training the Model

Run the following command from the project root:

```bash
python train.py
```

This will:
- Train the model on the training set
- Validate on the validation set
- Save the final model as `segmentation_model_full.pth`

---

## 9. Test-Time Inference

To generate predictions for the test dataset:

```bash
python predict_test.py
```

This script:
- Loads the trained model
- Runs inference on test images
- Saves predicted masks to `test_predictions/`

No metrics are computed on the test set.

---

## 10. Visualization

To visualize predictions on a single test image:

```bash
python visualise.py
```

This displays:
- Original image
- Predicted segmentation mask
- Overlay of prediction on the image

This file is for **qualitative demonstration only**.

---

## 11. Dependencies

Install required packages using:

```bash
pip install torch torchvision opencv-python numpy matplotlib
```


---

## 12. Key Notes

- Test data was **never used** for training or evaluation.
- All reported metrics are from the **validation set**.
- The project focuses on a **correct and clean ML pipeline**.

---

## 13. Conclusion

This project demonstrates a complete semantic segmentation workflow:
- Data loading and preprocessing
- Model training and validation
- Metric-based evaluation (IoU)
- Test-time inference
- Qualitative visualization

The implementation is modular, reproducible, and compliant with standard ML evaluation practices.

