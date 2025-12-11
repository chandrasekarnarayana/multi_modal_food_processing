"""Evaluation script for the multi-modal fusion model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import config
from src.datasets import create_dataloaders
from src.fusion_model import MultiModalFusionModel


def evaluate_model(args: argparse.Namespace) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Build test loader
    _, _, test_loader = create_dataloaders(
        batch_size=config.BATCH_SIZE,
        data_dir=args.data_dir,
        use_microscopy=args.use_microscopy,
        num_workers=0,
        device="cpu",
    )

    # Load model
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = MultiModalFusionModel(
        use_microscopy=args.use_microscopy,
        num_classes=config.NUM_CLASSES,
        num_reg_targets=config.NUM_REG_TARGETS,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_reg_preds: List[np.ndarray] = []
    all_reg_targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch)
            logits = outputs["logits_class"]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch["y_class"].cpu().numpy().tolist())
            all_reg_preds.extend(outputs["preds_reg"].cpu().numpy())
            all_reg_targets.extend(batch["y_reg"].cpu().numpy())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    reg_preds_arr = np.vstack(all_reg_preds)
    reg_targets_arr = np.vstack(all_reg_targets)

    # Metrics
    acc = (all_preds_arr == all_targets_arr).mean()
    mae = np.abs(reg_preds_arr - reg_targets_arr).mean()
    rmse = np.sqrt(((reg_preds_arr - reg_targets_arr) ** 2).mean())
    print(f"Test accuracy: {acc:.3f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_targets_arr, all_preds_arr, labels=list(range(config.NUM_CLASSES)))
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(config.NUM_CLASSES)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_path = fig_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # Save predictions CSV
    pred_dir = Path("results")
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "y_class_true": all_targets_arr,
            "y_class_pred": all_preds_arr,
        }
    )
    for i in range(reg_preds_arr.shape[1]):
        df[f"y_reg_true_{i}"] = reg_targets_arr[:, i]
        df[f"y_reg_pred_{i}"] = reg_preds_arr[:, i]

    csv_path = pred_dir / "predictions_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-modal fusion model")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--use_microscopy", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
