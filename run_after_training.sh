#!/usr/bin/env bash
# Resume pipeline after training is done. Run from repo root.
# Usage: bash run_after_training.sh

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

CKPT="${1:-results/models/best_model_fusion.pt}"

echo "=== Evaluation on test set (checkpoint: $CKPT) ==="
python -m src.evaluate --checkpoint_path "$CKPT" --device auto

echo "=== Visualization (UMAP + examples) ==="
python notebooks/02_model_training.py

echo "=== Active learning: uncertainty + random baselines ==="
python -m src.active_learning --device auto --n_rounds 7 --epochs_per_round 5 --batch_size 64 --acquire_per_round 50 --mc_passes 5

echo "=== Active learning overlay figure ==="
python notebooks/03_active_learning_demo.py

echo "=== Summaries (tables for README) ==="
python notebooks/04_results_summary.py

echo "Done. Check results/ for updated metrics, figures, and summaries."
