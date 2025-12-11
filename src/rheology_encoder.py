"""Rheology encoder using a 1D CNN + GRU hybrid."""

from __future__ import annotations

import torch
from torch import nn


class RheologyEncoder(nn.Module):
    """
    Encodes rheology curves into a fixed 64-D embedding using Conv1d + GRU with dropout.
    """

    def __init__(self, embed_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(input_size=64, hidden_size=embed_dim, batch_first=True)
        self.proj = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, rheo: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        rheo : torch.Tensor
            Tensor of shape (B, T_rheo) or (B, T_rheo, 1).
        Returns
        -------
        torch.Tensor
            Embedding of shape (B, embed_dim).
        """
        if rheo.dim() == 2:
            rheo = rheo.unsqueeze(-1)
        x = rheo.transpose(1, 2)  # (B, 1, T)
        x = self.conv(x)  # (B, C, T')
        x = x.transpose(1, 2)  # (B, T', C)
        _, h_n = self.gru(x)
        out = self.proj(h_n.squeeze(0))
        return out


if __name__ == "__main__":
    encoder = RheologyEncoder()
    dummy = torch.randn(4, 100)
    out = encoder(dummy)
    print("Output shape:", out.shape)
