import torch


def compute_average_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    return total_loss / len(loader)


def compute_iou(model, loader, num_classes, device):
    model.eval()
    iou = torch.zeros(num_classes)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = torch.argmax(model(images), dim=1)

            for cls in range(num_classes):
                inter = ((preds == cls) & (masks == cls)).sum().float()
                union = ((preds == cls) | (masks == cls)).sum().float()

                if union > 0:
                    iou[cls] += inter / union

    return iou / len(loader)
