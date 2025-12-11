"""Generic utilities for data handling and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from src.config import config


# -----------------------------
# Filesystem helpers
# -----------------------------

def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory (and parents) if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The resolved directory path.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_array(name: str, array: np.ndarray, base_dir: str | Path | None = None) -> Path:
    """
    Save a numpy array to the processed data directory as .npy.

    Parameters
    ----------
    name:
        Filename without extension (e.g., "videos_train").
    array:
        Numpy array to save.
    base_dir:
        Optional override for the output directory. Defaults to config.DATA_PROCESSED_DIR.
    """
    target_dir = Path(base_dir) if base_dir is not None else Path(config.DATA_PROCESSED_DIR)
    ensure_dir(target_dir)
    path = target_dir / f"{name}.npy"
    np.save(path, array)
    return path


def load_array(name: str, base_dir: str | Path | None = None) -> np.ndarray:
    """
    Load a numpy array from the processed data directory.

    Parameters
    ----------
    name:
        Filename without extension (e.g., "videos_train").
    base_dir:
        Optional override for the input directory. Defaults to config.DATA_PROCESSED_DIR.
    """
    target_dir = Path(base_dir) if base_dir is not None else Path(config.DATA_PROCESSED_DIR)
    path = target_dir / f"{name}.npy"
    return np.load(path)


# -----------------------------
# Normalization utilities
# -----------------------------

def normalize_video(video: np.ndarray) -> np.ndarray:
    """
    Normalize a video to [0, 1] range per sample.

    Parameters
    ----------
    video:
        Array shaped (T, H, W, C) or (N, T, H, W, C).
    """
    video = video.astype(np.float32)
    if video.ndim == 4:
        data = video
        min_val = data.min()
        max_val = data.max()
        return (data - min_val) / (max_val - min_val + 1e-8)
    if video.ndim == 5:
        data = video
        min_val = data.min(axis=(1, 2, 3, 4), keepdims=True)
        max_val = data.max(axis=(1, 2, 3, 4), keepdims=True)
        return (data - min_val) / (max_val - min_val + 1e-8)
    raise ValueError("Expected video with 4 or 5 dimensions (T,H,W,C) or (N,T,H,W,C)")


def normalize_rheology(
    curves: np.ndarray, method: Literal["zscore", "minmax"] = "zscore"
) -> np.ndarray:
    """
    Normalize rheology curves (N, T).

    Parameters
    ----------
    curves:
        Array of rheology measurements with shape (N, T) or (T,).
    method:
        \"zscore\" for zero-mean/unit-std per sample, \"minmax\" for [0, 1] scaling per sample.
    """
    data = np.asarray(curves, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]

    if method == "zscore":
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        normalized = (data - mean) / std
    elif method == "minmax":
        min_val = data.min(axis=1, keepdims=True)
        max_val = data.max(axis=1, keepdims=True)
        span = max_val - min_val
        span[span == 0] = 1.0
        normalized = (data - min_val) / span
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'zscore' or 'minmax'.")

    return normalized if curves.ndim > 1 else normalized.squeeze(0)


__all__ = [
    "config",
    "ensure_dir",
    "save_array",
    "load_array",
    "normalize_video",
    "normalize_rheology",
]
