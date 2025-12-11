"""Synthetic multi-modal time-series data generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from src.config import config
from src.data_utils import normalize_rheology, normalize_video


@dataclass
class RheologyParams:
    """Rheology parameter container."""

    base_viscosity: float
    decay_rate: float

    def as_array(self) -> np.ndarray:
        return np.array([self.base_viscosity, self.decay_rate], dtype=np.float32)


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_synthetic_rheology_parameters(
    n_samples: int, num_classes: int = 3, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample rheology parameters conditioned on viscosity class (balanced) with mild variation.

    Returns
    -------
    params : np.ndarray
        Array of shape (N, 2) with [base_viscosity, decay_rate].
    labels_class : np.ndarray
        Integer class labels (0=low, 1=medium, 2=high viscosity).
    """
    rng = _rng(seed)
    # Balanced classes
    base_labels = np.arange(num_classes)
    labels = np.tile(base_labels, int(np.ceil(n_samples / num_classes)))[:n_samples]
    rng.shuffle(labels)
    params = np.zeros((n_samples, 2), dtype=np.float32)

    # Class prototypes (base_viscosity [Pa·s], decay_rate [1/s])
    prototypes = {
        0: (0.5, 3.5),  # low viscosity, fast decay
        1: (1.5, 2.0),  # medium viscosity, moderate decay
        2: (3.5, 0.8),  # high viscosity, slow decay
    }
    noise_scale = (0.1, 0.3)  # Gaussian variation per class

    for cls in range(num_classes):
        mask = labels == cls
        if not np.any(mask):
            continue
        base_proto, decay_proto = prototypes[cls]
        base = rng.normal(loc=base_proto, scale=noise_scale[0], size=mask.sum())
        decay = rng.normal(loc=decay_proto, scale=noise_scale[1], size=mask.sum())
        params[mask, 0] = np.clip(base, 0.2, None)
        params[mask, 1] = np.clip(decay, 0.2, None)

    return params, labels.astype(np.int64)


def generate_synthetic_rheology_curves(
    params: np.ndarray,
    time_steps: int,
    material_identity: np.ndarray,
    noise_level: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate rheology curves with shear-thinning, temperature drift, hysteresis, and sensor noise.
    """
    rng = _rng(seed)
    params = np.asarray(params, dtype=np.float32)
    t = np.linspace(0, 1, time_steps, dtype=np.float32)
    shear_rate = np.logspace(-1, 2, time_steps, dtype=np.float32)

    base = params[:, 0:1]
    decay = params[:, 1:2]
    identity_mod = 0.1 * (material_identity @ np.ones((material_identity.shape[1], 1))).reshape(-1, 1)

    stretch = 1.1
    shear_thinning = (base + identity_mod) * (shear_rate[None, :] ** (-0.45)) * np.exp(
        -decay * (t[None, :] ** stretch)
    )
    temp_drift = 0.03 * base * (1 + np.sin(2 * np.pi * t)[None, :])
    drift_noise = noise_level * np.cumsum(rng.normal(size=shear_thinning.shape), axis=1) / time_steps

    # Hysteresis/overshoot around mid-shear transitions
    hysteresis = 0.05 * base * np.exp(-((t - 0.5) ** 2) / 0.02)

    signal = shear_thinning + temp_drift + drift_noise + hysteresis
    signal = np.maximum(signal, 0.0)
    return signal


def _make_blob_frame(
    height: int, width: int, centers: np.ndarray, amplitudes: np.ndarray, sigma: float
) -> np.ndarray:
    """Render a frame with Gaussian blobs."""
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]
    frame = np.zeros((height, width), dtype=np.float32)
    for (cy, cx), amp in zip(centers, amplitudes):
        frame += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return frame


def generate_synthetic_videos(
    params: np.ndarray,
    material_identity: np.ndarray,
    num_frames: int,
    height: int,
    width: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate videos with motion tied to rheology parameters and material identity.
    Adds motion blur, illumination gradients, temporal jitter, occlusions, and frame drops.
    """
    rng = _rng(seed)
    n_samples = params.shape[0]
    videos = np.zeros((n_samples, num_frames, height, width, 1), dtype=np.float32)

    for i in range(n_samples):
        base_visc, decay_rate = params[i]
        identity_mod = 1.0 + 0.2 * (material_identity[i].mean())
        # Strong dependency: low viscosity -> higher motion
        motion_scale = 2.0 / (base_visc * identity_mod + 0.1)
        sigma = max(1.5, 3.5 - decay_rate)
        n_blobs = np.random.default_rng(rng.integers(0, 1_000_000)).integers(5, 9)
        centers = rng.uniform([0, 0], [height, width], size=(n_blobs, 2))
        amplitudes = rng.uniform(0.3, 1.0, size=n_blobs)

        drop_count = rng.choice([0, 1, 2], p=[0.85, 0.1, 0.05])
        drop_indices = set(rng.choice(num_frames, size=drop_count, replace=False).tolist())

        for t_idx in range(num_frames):
            if t_idx in drop_indices and t_idx > 0:
                videos[i, t_idx] = videos[i, t_idx - 1]
                continue

            shift = rng.normal(scale=motion_scale * 3.0, size=centers.shape)
            centers = np.clip(centers + shift, [0, 0], [height - 1, width - 1])
            frame = _make_blob_frame(height, width, centers, amplitudes, sigma=sigma)

            grad_x = np.linspace(0.8, 1.1, width)
            grad_y = np.linspace(0.8, 1.05, height)[:, None]
            frame *= grad_y * grad_x

            if rng.random() < 0.1:
                frame += rng.normal(scale=0.1, size=frame.shape)

            blur_sigma = max(0.5, 2.0 / (motion_scale + 0.5))
            frame = gaussian_filter(frame, sigma=blur_sigma)

            # Vignetting and occlusions (e.g., bubbles)
            vignette = np.outer(np.linspace(0.9, 1.0, height), np.linspace(0.9, 1.0, width))
            frame *= vignette
            if rng.random() < 0.2:
                for _ in range(rng.integers(1, 3)):
                    cy, cx = rng.integers(0, height), rng.integers(0, width)
                    r = rng.integers(height // 20, height // 10)
                    y, x = np.ogrid[-cy : height - cy, -cx : width - cx]
                    mask = x * x + y * y <= r * r
                    frame[mask] *= rng.uniform(0.2, 0.6)

            frame = np.clip(frame, 0, None)
            videos[i, t_idx, :, :, 0] = frame

    return videos


def generate_synthetic_microscopy_images(
    params: np.ndarray, height: int, width: int, seed: Optional[int] = None
) -> np.ndarray:
    """
    Higher-resolution microscopy-like images influenced by viscosity parameters with realistic noise.
    """
    rng = _rng(seed)
    n_samples = params.shape[0]
    images = np.zeros((n_samples, height, width, 1), dtype=np.float32)

    for i in range(n_samples):
        base_visc, decay_rate = params[i]
        cls_intensity = np.clip(base_visc / 2.5, 0.2, 1.0)
        texture_scale = max(1.0, 5.0 - decay_rate)
        grid_y, grid_x = np.mgrid[0:height, 0:width]

        if base_visc < 0.9:
            pattern = np.sin(grid_x / texture_scale) * np.cos(grid_y / (texture_scale * 0.5))
        elif base_visc < 1.6:
            pattern = np.sin(grid_x / (texture_scale * 0.8)) + np.sin(grid_y / (texture_scale * 0.8))
        else:
            pattern = gaussian_filter(
                rng.normal(size=(height, width)), sigma=texture_scale * 0.3
            )

        illum = 0.7 + 0.3 * np.sin(np.pi * grid_x / width) * np.cos(np.pi * grid_y / height)
        img = cls_intensity * pattern * illum

        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        shot = rng.poisson(lam=img_norm * 20) / 20.0
        gaussian = rng.normal(scale=0.05, size=img.shape)
        speckle = img_norm * rng.normal(scale=0.1, size=img.shape)
        img_final = np.clip(shot + gaussian + speckle, 0, 1)

        # Vignetting
        vignette = np.outer(np.linspace(0.85, 1.0, height), np.linspace(0.85, 1.0, width))
        img_final *= vignette
        images[i, :, :, 0] = img_final

    return images


def generate_full_synthetic_dataset(
    n_samples: int,
    cfg=config,
    include_microscopy: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Generate coupled videos, rheology curves, optional microscopy, and labels.
    """
    rng = _rng(seed)
    identity_dim = 8
    # Batch effects (instrument/day)
    batch_ids = rng.integers(0, 5, size=n_samples)
    batch_base_offset = rng.normal(scale=0.1, size=5)
    batch_decay_offset = rng.normal(scale=0.05, size=5)

    params, labels_class = generate_synthetic_rheology_parameters(
        n_samples, num_classes=cfg.NUM_CLASSES, seed=rng.integers(0, 1_000_000)
    )
    # Apply batch offsets to parameters
    params[:, 0] = params[:, 0] + batch_base_offset[batch_ids]
    params[:, 1] = params[:, 1] + batch_decay_offset[batch_ids]

    material_identity = rng.normal(size=(n_samples, identity_dim)).astype(np.float32)
    rheology = generate_synthetic_rheology_curves(
        params,
        time_steps=cfg.RHEOLOGY_TIME_STEPS,
        material_identity=material_identity,
        seed=rng.integers(0, 1_000_000),
    )
    videos = generate_synthetic_videos(
        params,
        material_identity,
        num_frames=cfg.VIDEO_NUM_FRAMES,
        height=cfg.VIDEO_HEIGHT,
        width=cfg.VIDEO_WIDTH,
        seed=rng.integers(0, 1_000_000),
    )
    microscopy = None
    if include_microscopy:
        microscopy = generate_synthetic_microscopy_images(
            params,
            height=max(cfg.VIDEO_HEIGHT, 128),
            width=max(cfg.VIDEO_WIDTH, 128),
            seed=rng.integers(0, 1_000_000),
        )

    # Normalize
    videos = normalize_video(videos)
    rheology = normalize_rheology(rheology, method="zscore")
    if microscopy is not None:
        microscopy = normalize_video(microscopy[:, None, :, :, :])[:, 0]  # reuse video normalizer

    # Missing modality simulation
    missing_mask = np.ones((n_samples, 2), dtype=bool)  # [video_present, microscopy_present]
    drop_prob = rng.uniform(0.1, 0.15)
    drop_video = rng.random(n_samples) < drop_prob
    drop_micro = rng.random(n_samples) < drop_prob if include_microscopy else np.zeros(n_samples, bool)

    for i in range(n_samples):
        if drop_video[i]:
            missing_mask[i, 0] = False
            videos[i] = 0.0
        if include_microscopy and drop_micro[i]:
            missing_mask[i, 1] = False
            microscopy[i] = 0.0

    metadata = []
    for i in range(n_samples):
        metadata.append(
            {
                "index": int(i),
                "viscosity_class": int(labels_class[i]),
                "batch_id": int(batch_ids[i]),
                "parameters": {
                    "base_viscosity": float(params[i, 0]),
                    "decay_rate": float(params[i, 1]),
                },
                "material_identity": material_identity[i].tolist(),
                "missing": {
                    "video": bool(missing_mask[i, 0] == 0),
                    "microscopy": bool(missing_mask[i, 1] == 0 if include_microscopy else False),
                },
            }
        )

    labels_reg = params.astype(np.float32)
    return videos, rheology, microscopy, labels_class.astype(np.int64), labels_reg, missing_mask, metadata


__all__ = [
    "RheologyParams",
    "generate_synthetic_rheology_parameters",
    "generate_synthetic_rheology_curves",
    "generate_synthetic_videos",
    "generate_synthetic_microscopy_images",
    "generate_full_synthetic_dataset",
]
