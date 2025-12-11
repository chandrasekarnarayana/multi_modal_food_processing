# file: notebooks/01_data_preparation.py  # to be converted to .ipynb
"""Synthetic data preparation for multi-modal time-series."""

from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt

from src.config import config
from src.data_utils import ensure_dir, save_array
from src.synthetic_data import generate_full_synthetic_dataset


# -----------------
# Configuration
# -----------------
N_SAMPLES = 900
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 123

processed_dir = ensure_dir(config.DATA_PROCESSED_DIR)

# -----------------
# Generate dataset
# -----------------
videos, rheology, microscopy, labels_class, labels_reg, missing_mask, metadata = (
    generate_full_synthetic_dataset(
        n_samples=N_SAMPLES,
        cfg=config,
        include_microscopy=True,
        seed=SEED,
    )
)

print("Shapes:")
print(" videos:", videos.shape)
print(" rheology:", rheology.shape)
print(" microscopy:", None if microscopy is None else microscopy.shape)
print(" labels_class:", labels_class.shape)
print(" labels_reg:", labels_reg.shape)


# -----------------
# Split helpers
# -----------------

def split_indices(n: int, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx


def save_split(split: str, indices: np.ndarray):
    save_array(f"videos_{split}", videos[indices])
    save_array(f"rheology_{split}", rheology[indices])
    if microscopy is not None:
        save_array(f"microscopy_{split}", microscopy[indices])
    save_array(f"labels_class_{split}", labels_class[indices])
    save_array(f"labels_reg_{split}", labels_reg[indices])
    save_array(f"missing_modalities_{split}", missing_mask[indices])
    meta_subset = [metadata[int(i)] for i in indices]
    meta_path = processed_dir / f"metadata_{split}.json"
    with meta_path.open("w") as f:
        json.dump(meta_subset, f, indent=2)
    print(f"Saved metadata -> {meta_path}")


train_idx, val_idx, test_idx = split_indices(N_SAMPLES, VAL_FRAC, TEST_FRAC, seed=SEED)

save_split("train", train_idx)
save_split("val", val_idx)
save_split("test", test_idx)


# -----------------
# Visualization
# -----------------

def plot_rheology_by_class(curves: np.ndarray, labels: np.ndarray, save_path: Path):
    t = np.linspace(0, 1, curves.shape[1])
    plt.figure(figsize=(10, 6))
    for cls in np.unique(labels):
        mask = labels == cls
        if not mask.any():
            continue
        idx = np.where(mask)[0][:30]  # show more per class
        for j in idx:
            plt.plot(t, curves[j], alpha=0.4, label=f"Class {cls}" if j == idx[0] else None)
    plt.xlabel("Time (normalized)")
    plt.ylabel("Normalized viscosity")
    plt.title("Rheology curves by class (v2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_video_frames_grid(videos_arr: np.ndarray, labels_arr: np.ndarray, save_path: Path):
    classes = np.unique(labels_arr)
    num_examples = 3
    num_frames_to_show = 6
    plt.figure(figsize=(num_frames_to_show * 2, len(classes) * 2.5))
    row = 0
    for cls in classes:
        cls_indices = np.where(labels_arr == cls)[0][:num_examples]
        for idx_offset, idx in enumerate(cls_indices):
            frames = videos_arr[idx][:num_frames_to_show]
            for col, fi in enumerate(range(num_frames_to_show)):
                plt.subplot(len(classes) * num_examples, num_frames_to_show, row * num_frames_to_show + col + 1)
                plt.imshow(frames[fi, :, :, 0], cmap="gray")
                plt.axis("off")
                if col == 0:
                    plt.title(f"Class {cls}")
            row += 1
    plt.suptitle("Example video frames by class (v2)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_microscopy_grid(images: np.ndarray, labels: np.ndarray, num_examples: int = 6):
    plt.figure(figsize=(10, 4))
    for i in range(num_examples):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.title(f"Class {labels[i]}")
    plt.suptitle("Microscopy-like images")
    plt.tight_layout()
    plt.show()


fig_dir = Path("results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

plot_rheology_by_class(rheology, labels_class, save_path=fig_dir / "rheology_by_class_v2.png")
plot_video_frames_grid(videos, labels_class, save_path=fig_dir / "example_video_frames_v2.png")
if microscopy is not None:
    plot_microscopy_grid(microscopy, labels_class)
