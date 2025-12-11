# file: notebooks/03_active_learning_demo.py  # to be converted to .ipynb
"""Active learning metrics visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

metrics_path = Path("results/active_learning_metrics.csv")
fig_dir = Path("results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

if not metrics_path.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

# Load metrics
metrics = pd.read_csv(metrics_path)
print("Active learning metrics:")
print(metrics)

# Plot accuracy vs labelled samples
plt.figure(figsize=(7, 5))
plt.plot(metrics["num_labelled"], metrics["val_accuracy"], marker="o")
plt.xlabel("Number of labelled samples")
plt.ylabel("Validation accuracy")
plt.title("Active Learning Curve (overlay)")
plt.grid(True, linestyle="--", alpha=0.5)
curve_path = fig_dir / "active_learning_curve_overlay.png"
plt.savefig(curve_path)
plt.close()
print(f"Saved overlay curve to {curve_path}")
