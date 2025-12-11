"""Video encoder combining stronger per-frame CNN features with temporal aggregation."""

from __future__ import annotations

import torch
from torch import nn


class VideoEncoder(nn.Module):
    """
    Encodes a sequence of video frames into a fixed 128-D embedding.

    Expects input of shape (B, T, C, H, W). Frames are encoded with a deeper
    CNN and aggregated temporally via a bidirectional GRU. Dropout is applied
    to reduce overfitting.
    """

    def __init__(self, embed_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # Assuming input 64x64 -> after 4 pools of /2 => 4x4
        self.frame_to_feat = nn.Sequential(
            nn.Linear(128 * 4 * 4, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        video : torch.Tensor
            Tensor of shape (B, T, C, H, W).
        Returns
        -------
        torch.Tensor
            Video embedding of shape (B, embed_dim).
        """
        b, t, c, h, w = video.shape
        frames = video.reshape(b * t, c, h, w)
        frame_feats = self.frame_encoder(frames)
        frame_feats = self.frame_to_feat(frame_feats)  # (B*T, embed_dim)
        frame_feats = frame_feats.view(b, t, -1)
        gru_out, _ = self.gru(frame_feats)  # (B, T, 2*embed_dim)
        # Use final time step representation
        fused = gru_out[:, -1, :]
        fused = self.proj(fused)
        return fused


if __name__ == "__main__":
    encoder = VideoEncoder()
    dummy = torch.randn(2, 20, 1, 64, 64)
    out = encoder(dummy)
    print("Output shape:", out.shape)
