"""PyTorch datasets and loaders for synthetic multi-modal time-series data."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MultiModalTimeSeriesDataset(Dataset):
    """
    Dataset for videos, rheology curves, optional microscopy, and labels.
    """

    def __init__(
        self,
        split: str,
        data_dir: str | Path = "data/processed",
        use_microscopy: bool = False,
        device: str | torch.device = "cpu",
        video_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        microscopy_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """
        Parameters
        ----------
        split:
            One of {"train", "val", "test"}.
        data_dir:
            Directory containing processed .npy files.
        use_microscopy:
            Whether to load microscopy data if available.
        device:
            Device to place tensors on.
        video_transform:
            Optional transform applied to the video tensor (T, C, H, W).
        microscopy_transform:
            Optional transform applied to microscopy tensor (C, H, W).
        """
        self.split = split
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)
        self.use_microscopy = use_microscopy
        self.video_transform = video_transform
        self.microscopy_transform = microscopy_transform

        self.videos = self._load_required(f"videos_{split}.npy")
        self.rheology = self._load_required(f"rheology_{split}.npy")
        self.labels_class = self._load_required(f"labels_class_{split}.npy").astype(np.int64)
        self.labels_reg = self._load_required(f"labels_reg_{split}.npy")

        self.microscopy = None
        if use_microscopy:
            mic_path = self.data_dir / f"microscopy_{split}.npy"
            if not mic_path.exists():
                raise FileNotFoundError(
                    f"Microscopy requested but file not found: {mic_path}. "
                    "Set use_microscopy=False or generate microscopy data."
                )
            self.microscopy = np.load(mic_path)

    def _load_required(self, filename: str) -> np.ndarray:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return np.load(path)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> dict:
        video = torch.as_tensor(self.videos[idx], dtype=torch.float32, device=self.device)
        video = video.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        if self.video_transform is not None:
            video = self.video_transform(video)

        rheology = torch.as_tensor(self.rheology[idx], dtype=torch.float32, device=self.device)

        if self.use_microscopy and self.microscopy is not None:
            microscopy = torch.as_tensor(
                self.microscopy[idx], dtype=torch.float32, device=self.device
            )
            microscopy = microscopy.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            if self.microscopy_transform is not None:
                microscopy = self.microscopy_transform(microscopy)
            microscopy_tensor = microscopy
        else:
            microscopy_tensor = torch.empty(0, device=self.device)

        y_class = torch.as_tensor(self.labels_class[idx], dtype=torch.long, device=self.device)
        y_reg = torch.as_tensor(self.labels_reg[idx], dtype=torch.float32, device=self.device)

        return {
            "video": video,
            "rheology": rheology,
            "microscopy": microscopy_tensor,
            "y_class": y_class,
            "y_reg": y_reg,
            "index": idx,
        }


def create_dataloaders(
    batch_size: int,
    data_dir: str | Path = "data/processed",
    use_microscopy: bool = False,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create train/val/test dataloaders.
    """
    splits = ["train", "val", "test"]
    datasets = {
        split: MultiModalTimeSeriesDataset(
            split=split,
            data_dir=data_dir,
            use_microscopy=use_microscopy,
            device=device,
        )
        for split in splits
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return loaders["train"], loaders["val"], loaders["test"]


__all__ = ["MultiModalTimeSeriesDataset", "create_dataloaders"]
