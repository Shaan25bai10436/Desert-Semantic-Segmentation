import torch.nn as nn


class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Deeper features
        self.middle = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        # Restore spatial size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU()
        )

        # Pixel-wise classification
        self.classifier = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return self.classifier(x)
