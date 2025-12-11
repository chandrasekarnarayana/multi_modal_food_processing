"""Multi-modal fusion model combining video, rheology, and optional microscopy encoders."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from src.video_encoder import VideoEncoder
from src.rheology_encoder import RheologyEncoder
from src.microscopy_encoder import MicroscopyEncoder


class MultiModalFusionModel(nn.Module):
    """
    Fuses video, rheology, and optional microscopy embeddings for classification and regression.
    Supports partial supervision and heterogeneous observation dropout (missing modalities).
    """

    def __init__(
        self,
        use_microscopy: bool = False,
        embed_dim_video: int = 128,
        embed_dim_rheo: int = 64,
        embed_dim_micro: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 3,
        num_reg_targets: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_microscopy = use_microscopy

        self.video_encoder = VideoEncoder(embed_dim=embed_dim_video)
        self.rheology_encoder = RheologyEncoder(embed_dim=embed_dim_rheo)
        self.microscopy_encoder = (
            MicroscopyEncoder(embed_dim=embed_dim_micro) if use_microscopy else None
        )

        # Missing-token embeddings for absent modalities
        self.missing_video = nn.Parameter(torch.zeros(embed_dim_video))
        self.missing_rheo = nn.Parameter(torch.zeros(embed_dim_rheo))
        self.missing_micro = nn.Parameter(torch.zeros(embed_dim_micro)) if use_microscopy else None

        # Gating for robust fusion under incomplete data
        self.gate_video = nn.Linear(embed_dim_video, embed_dim_video)
        self.gate_rheo = nn.Linear(embed_dim_rheo, embed_dim_rheo)
        self.gate_micro = nn.Linear(embed_dim_micro, embed_dim_micro) if use_microscopy else None

        # Project all modalities to a common dim for attention
        self.proj_video = nn.Linear(embed_dim_video, hidden_dim)
        self.proj_rheo = nn.Linear(embed_dim_rheo, hidden_dim)
        self.proj_micro = nn.Linear(embed_dim_micro, hidden_dim) if use_microscopy else None

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.fusion_dropout = nn.Dropout(dropout)

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_reg_targets),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-modal fusion.

        Parameters
        ----------
        batch:
            Dictionary containing modality tensors and labels.

        Returns
        -------
        dict
            Contains logits, regression preds, and intermediate embeddings.
        """
        video = batch["video"]
        rheology = batch["rheology"]
        microscopy = batch.get("microscopy", None)
        missing = batch.get("missing_modalities", None)

        # Simulate modality dropout during training (heterogeneous observation dropout)
        if self.training:
            dropout_p = 0.1
            if torch.rand(1).item() < dropout_p:
                missing = missing or {}
                missing["video"] = True
            if torch.rand(1).item() < dropout_p:
                missing = missing or {}
                missing["rheology"] = True
            if self.use_microscopy and torch.rand(1).item() < dropout_p:
                missing = missing or {}
                missing["microscopy"] = True

        embed_video = self.video_encoder(video)
        embed_rheo = self.rheology_encoder(rheology)

        embeds = []
        modality_weights = {}

        video_present = True
        rheo_present = True
        micro_present = self.use_microscopy and microscopy is not None and torch.is_tensor(microscopy) and microscopy.numel() > 0

        if missing:
            video_present = not missing.get("video", False)
            rheo_present = not missing.get("rheology", False)
            micro_present = micro_present and not missing.get("microscopy", False)

        if video_present:
            v_gate = torch.sigmoid(self.gate_video(embed_video))
            v_emb = v_gate * embed_video
            modality_weights["video"] = v_gate
        else:
            v_emb = self.missing_video.unsqueeze(0).expand(embed_video.size(0), -1)
            modality_weights["video"] = torch.zeros_like(embed_video)

        if rheo_present:
            r_gate = torch.sigmoid(self.gate_rheo(embed_rheo))
            r_emb = r_gate * embed_rheo
            modality_weights["rheology"] = r_gate
        else:
            r_emb = self.missing_rheo.unsqueeze(0).expand(embed_rheo.size(0), -1)
            modality_weights["rheology"] = torch.zeros_like(embed_rheo)

        embed_micro = None
        if self.use_microscopy:
            if micro_present:
                embed_micro = self.microscopy_encoder(microscopy)
                m_gate = torch.sigmoid(self.gate_micro(embed_micro))
                m_emb = m_gate * embed_micro
                modality_weights["microscopy"] = m_gate
            else:
                m_emb = self.missing_micro.unsqueeze(0).expand(embed_video.size(0), -1)
                modality_weights["microscopy"] = torch.zeros_like(embed_video)
        else:
            m_emb = None

        # Project to common dim and stack as sequence
        seq_list = [
            self.proj_video(v_emb),
            self.proj_rheo(r_emb),
        ]
        if self.use_microscopy:
            seq_list.append(self.proj_micro(m_emb))
        seq = torch.stack(seq_list, dim=1)  # (B, M, hidden_dim)

        attn_out, attn_weights = self.attn(seq, seq, seq)
        # Pool (mean) over modalities
        hidden = self.fusion_dropout(attn_out.mean(dim=1))

        logits_class = self.classification_head(hidden)
        preds_reg = self.regression_head(hidden)

        out = {
            "logits_class": logits_class,
            "preds_reg": preds_reg,
            "embed_video": embed_video,
            "embed_rheo": embed_rheo,
            "embed_fused": hidden,
            "modality_weights": modality_weights,
            "attn_weights": attn_weights,
            "modalities_present": {
                "video": video_present,
                "rheology": rheo_present,
                "microscopy": micro_present if self.use_microscopy else False,
            },
        }
        if embed_micro is not None:
            out["embed_micro"] = embed_micro
        return out


def compute_losses(
    batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], alpha_reg: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute classification and regression losses and return combined loss.

    Parameters
    ----------
    batch:
        Batch dict containing ground-truth labels.
    outputs:
        Model outputs containing logits_class and preds_reg.
    alpha_reg:
        Scaling factor for regression loss.
    """
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    class_loss = ce_loss(outputs["logits_class"], batch["y_class"])
    reg_loss = mse_loss(outputs["preds_reg"], batch["y_reg"])
    total_loss = class_loss + alpha_reg * reg_loss
    return total_loss, class_loss, reg_loss


if __name__ == "__main__":
    # Simple sanity check
    batch_size = 4
    T_video = 20
    H = W = 64
    T_rheo = 100
    num_classes = 3
    num_reg_targets = 2

    batch = {
        "video": torch.randn(batch_size, T_video, 1, H, W),
        "rheology": torch.randn(batch_size, T_rheo),
        "microscopy": torch.randn(batch_size, 1, H, W),
        "y_class": torch.randint(0, num_classes, (batch_size,)),
        "y_reg": torch.randn(batch_size, num_reg_targets),
    }

    model = MultiModalFusionModel(use_microscopy=True, num_classes=num_classes, num_reg_targets=num_reg_targets)
    outputs = model(batch)
    total_loss, class_loss, reg_loss = compute_losses(batch, outputs, alpha_reg=0.5)

    print("logits_class shape:", outputs["logits_class"].shape)
    print("preds_reg shape:", outputs["preds_reg"].shape)
    print("Total loss:", total_loss.item())
