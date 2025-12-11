# file: notebooks/02_model_training.py  # to be converted to .ipynb
"""Analyze trained model embeddings and visualize modalities."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import config
from src.datasets import MultiModalTimeSeriesDataset
from src.fusion_model import MultiModalFusionModel
from src.utils_visualization import (
    plot_embeddings_umap,
    plot_rheology_curves_by_class,
    plot_video_examples,
)


# -----------------------------
# Settings
# -----------------------------
CHECKPOINT_PATH = Path("results/models/best_model.pt")
DATA_DIR = Path("data/processed")
USE_MICROSCOPY = False
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load model and data
# -----------------------------
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model = MultiModalFusionModel(
    use_microscopy=USE_MICROSCOPY,
    num_classes=config.NUM_CLASSES,
    num_reg_targets=config.NUM_REG_TARGETS,
).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Use test split
dataset = MultiModalTimeSeriesDataset(
    split="test",
    data_dir=DATA_DIR,
    use_microscopy=USE_MICROSCOPY,
    device="cpu",
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------
# Collect embeddings and labels
# -----------------------------
all_fused = []
all_labels = []
all_videos = []
all_rheo = []

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(batch)
        all_fused.append(outputs["embed_fused"].cpu().numpy())
        all_labels.append(batch["y_class"].cpu().numpy())
        all_videos.append(batch["video"].cpu().numpy())
        all_rheo.append(batch["rheology"].cpu().numpy())

embeddings = np.vstack(all_fused)
labels = np.concatenate(all_labels)
videos = np.concatenate(all_videos)
rheology = np.concatenate(all_rheo)

# -----------------------------
# Visualizations
# -----------------------------
fig_dir = Path("results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

plot_embeddings_umap(
    embeddings=embeddings,
    labels=labels,
    save_path=fig_dir / "fused_embeddings_umap.png",
    title="UMAP of fused embeddings",
)

plot_video_examples(
    videos=videos,
    labels=labels,
    save_path=fig_dir / "example_video_frames.png",
    num_examples=3,
    num_frames_to_show=5,
    random_state=0,
)

plot_rheology_curves_by_class(
    rheology=rheology,
    labels=labels,
    save_path=fig_dir / "example_rheology_curves.png",
    max_per_class=20,
)

print("Saved UMAP, video examples, and rheology curve plots to", fig_dir)
