"""Compact two-band radar CNN using supported PyTorch components."""

import torch
from torch import nn


def conv_block(inputs: int, filters: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, filters, 3, padding=1, bias=False),
        nn.BatchNorm2d(filters),
        nn.SiLU(),
        nn.Conv2d(filters, filters, 3, padding=1, bias=False),
        nn.BatchNorm2d(filters),
        nn.SiLU(),
    )


class IcebergCNN(nn.Module):
    """Downsample radar features, optionally adding standardized angle and missingness."""

    def __init__(
        self, filters: int = 16, hidden: int = 64, *, use_angle: bool = False, dropout: float = 0.3
    ):
        super().__init__()
        self.use_angle = use_angle
        stages = []
        inputs = 2
        for stage, width in enumerate([filters, filters * 2, filters * 4, filters * 8]):
            stages.append(conv_block(inputs, width))
            if stage < 3:
                stages.append(nn.MaxPool2d(2))
            inputs = width
        self.features = nn.Sequential(*stages)
        self.angle_encoder = nn.Sequential(nn.Linear(2, 16), nn.SiLU()) if use_angle else None
        self.head = nn.Sequential(
            nn.Linear(filters * 16 + (16 if use_angle else 0), hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, images, angles):
        features = self.features(images)
        pooled = [features.mean(dim=(-2, -1)), features.amax(dim=(-2, -1))]
        if self.angle_encoder is not None:
            pooled.append(self.angle_encoder(angles))
        return self.head(torch.cat(pooled, dim=1)).squeeze(1)
