"""Microscopy encoder using a small CNN."""

from __future__ import annotations

import torch
from torch import nn


class MicroscopyEncoder(nn.Module):
    """
    Encodes microscopy images into a fixed 64-D embedding.
    """

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Linear(64 * 4 * 4, embed_dim)

    def forward(self, microscopy: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        microscopy : torch.Tensor
            Tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Embedding of shape (B, embed_dim).
        """
        feats = self.features(microscopy)
        return self.head(feats)


if __name__ == "__main__":
    encoder = MicroscopyEncoder()
    dummy = torch.randn(3, 1, 64, 64)
    out = encoder(dummy)
    print("Output shape:", out.shape)
