"""
Embedding network to produce fixed-size vectors from spectrograms.
"""
import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self, in_channels: int = 1, embedding_dim: int = 128, base_filters: int = 16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(base_filters * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
