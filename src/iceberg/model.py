"""Port of legacy/scripts/models.py; cross-framework equality is unverified."""

import torch
from torch import nn


class LegacyBatchNorm(nn.Module):
    """Population variance and momentum=0 semantics of the original Keras layers.

    Native PyTorch batch norm updates running variance differently. Keeping the
    update explicit also supports a final minibatch containing one angle value.
    """

    def __init__(self, features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))

    def forward(self, values):
        axes = (0, *range(2, values.ndim))
        shape = [1, -1, *([1] * (values.ndim - 2))]
        if self.training:
            mean = values.mean(axes)
            variance = values.var(axes, correction=0)
            with torch.no_grad():
                self.running_mean.copy_(mean)
                self.running_var.copy_(variance)
        else:
            mean, variance = self.running_mean, self.running_var
        normalized = (values - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + 1e-3)
        return normalized * self.weight.reshape(shape) + self.bias.reshape(shape)


def conv_block(inputs: int, filters: int, count: int) -> nn.Sequential:
    layers = []
    for _ in range(count):
        conv = nn.Conv2d(inputs, filters, 3, padding=1)
        nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
        nn.init.zeros_(conv.bias)
        layers.extend([conv, nn.ELU()])
        inputs = filters
    return nn.Sequential(*layers)


def dense_block(inputs: int, hidden: int) -> nn.Sequential:
    linear = nn.Linear(inputs, hidden)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)
    return nn.Sequential(linear, nn.ELU(), LegacyBatchNorm(hidden), nn.Dropout(0.5))


class IcebergCNN(nn.Module):
    def __init__(self, filters: int = 8, hidden: int = 128):
        super().__init__()
        self.image_norm = LegacyBatchNorm(2)
        self.angle_norm = LegacyBatchNorm(1)
        self.pooled_branch = nn.Sequential(
            conv_block(2, filters, 3),
            LegacyBatchNorm(filters),
            nn.MaxPool2d(3, 3),
            nn.Dropout(0.2),
            conv_block(filters, filters * 2, 3),
            LegacyBatchNorm(filters * 2),
            nn.MaxPool2d(3, 2),
            nn.Dropout(0.2),
            conv_block(filters * 2, filters * 4, 3),
            LegacyBatchNorm(filters * 4),
            nn.MaxPool2d(3, 3),
            nn.Dropout(0.2),
            conv_block(filters * 4, filters * 8, 3),
            nn.Dropout(0.2),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            LegacyBatchNorm(filters * 8),
        )
        self.full_resolution_branch = nn.Sequential(
            conv_block(2, filters * 8, 6),
            nn.Dropout(0.2),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            LegacyBatchNorm(filters * 8),
        )
        self.head = nn.Sequential(
            dense_block(filters * 16 + 1, hidden),
            dense_block(hidden, hidden),
            nn.Linear(hidden, 1),
        )
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, images, angles):
        images = self.image_norm(images)
        joined = torch.cat(
            [
                self.pooled_branch(images),
                self.full_resolution_branch(images),
                self.angle_norm(angles),
            ],
            dim=1,
        )
        # BCEWithLogitsLoss combines the original sigmoid and BCE more stably.
        return self.head(joined).squeeze(1)
