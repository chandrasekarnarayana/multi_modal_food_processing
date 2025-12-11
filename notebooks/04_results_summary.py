# file: notebooks/04_results_summary.py
"""Summarize training ablations and active learning baselines."""

from pathlib import Path

import pandas as pd
import numpy as np

# Paths
results_dir = Path("results")

# Training metrics
train_files = {
    "fusion": results_dir / "training_metrics_fusion.csv",
    "video_only": results_dir / "training_metrics_video_only.csv",
    "rheology_only": results_dir / "training_metrics_rheo_only.csv",
}

summary_rows = []
for name, path in train_files.items():
    if path.exists():
        df = pd.read_csv(path)
        best = df["val_acc"].max()
        summary_rows.append({"mode": name, "best_val_acc": best})
    else:
        summary_rows.append({"mode": name, "best_val_acc": None})
train_summary = pd.DataFrame(summary_rows)

# Test metrics
preds_path = results_dir / "predictions_test.csv"
if preds_path.exists():
    preds = pd.read_csv(preds_path)
    test_acc = (preds["y_class_true"] == preds["y_class_pred"]).mean()
    reg_mae = np.abs(preds.filter(like="y_reg_pred").values - preds.filter(like="y_reg_true").values).mean()
    reg_rmse = np.sqrt(((preds.filter(like="y_reg_pred").values - preds.filter(like="y_reg_true").values) ** 2).mean())
else:
    test_acc = reg_mae = reg_rmse = None

# Active learning
al_unc_path = results_dir / "active_learning_uncertainty.csv"
al_rand_path = results_dir / "active_learning_random.csv"

al_rows = []
for name, path in [("uncertainty", al_unc_path), ("random", al_rand_path)]:
    if path.exists():
        df = pd.read_csv(path)
        first_acc = df["val_accuracy"].iloc[0]
        last_acc = df["val_accuracy"].iloc[-1]
        first_lab = df["num_labelled"].iloc[0]
        last_lab = df["num_labelled"].iloc[-1]
        al_rows.append(
            {
                "strategy": name,
                "first_acc": first_acc,
                "last_acc": last_acc,
                "first_labelled": first_lab,
                "last_labelled": last_lab,
            }
        )
    else:
        al_rows.append({"strategy": name, "first_acc": None, "last_acc": None, "first_labelled": None, "last_labelled": None})
al_summary = pd.DataFrame(al_rows)

# Write summary markdown
summary_md = ["# Results Summary", ""]
summary_md.append("## Training Ablations")
summary_md.append(train_summary.to_markdown(index=False))
summary_md.append("")
summary_md.append("## Test Metrics")
summary_md.append(f"Test accuracy: {test_acc}\n\nMAE: {reg_mae}\n\nRMSE: {reg_rmse}")
summary_md.append("")
summary_md.append("## Active Learning (Uncertainty vs Random)")
summary_md.append(al_summary.to_markdown(index=False))

out_path = results_dir / "results_summary.md"
out_path.write_text("\n".join(summary_md))
print(f"Saved summary to {out_path}")
print("\n".join(summary_md))
