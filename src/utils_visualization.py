"""Visualization utilities for embeddings, videos, and rheology curves."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import umap  # type: ignore


def plot_embeddings_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    title: str = "UMAP of fused embeddings",
    random_state: int = 42,
) -> None:
    """
    Reduce embeddings to 2D with UMAP and save a scatter plot.
    """
    reducer = umap.UMAP(random_state=random_state)
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", s=12, alpha=0.8)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_video_examples(
    videos: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    num_examples: int = 3,
    num_frames_to_show: int = 5,
    random_state: Optional[int] = None,
) -> None:
    """
    Plot rows of frames for randomly chosen videos.
    """
    rng = random.Random(random_state)
    indices = rng.sample(range(len(videos)), k=min(num_examples, len(videos)))
    plt.figure(figsize=(num_frames_to_show * 2.0, num_examples * 2.0))
    for row, idx in enumerate(indices):
        frames = videos[idx][:num_frames_to_show]
        for col, frame in enumerate(frames):
            plt.subplot(num_examples, num_frames_to_show, row * num_frames_to_show + col + 1)
            plt.imshow(frame[:, :, 0], cmap="gray")
            plt.axis("off")
            if col == 0:
                plt.title(f"Class {labels[idx]}")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_rheology_curves_by_class(
    rheology: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    max_per_class: int = 20,
) -> None:
    """
    Plot rheology curves separated by class.
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4), sharey=True)
    if n_classes == 1:
        axes = [axes]
    t = np.linspace(0, 1, rheology.shape[1])
    for ax, cls in zip(axes, unique_labels):
        mask = np.where(labels == cls)[0][:max_per_class]
        for idx in mask:
            ax.plot(t, rheology[idx], alpha=0.4)
        ax.set_title(f"Class {cls}")
        ax.set_xlabel("Time (normalized)")
    axes[0].set_ylabel("Normalized viscosity")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


__all__ = [
    "plot_embeddings_umap",
    "plot_video_examples",
    "plot_rheology_curves_by_class",
]
