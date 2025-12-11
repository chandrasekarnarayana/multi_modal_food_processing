#!/usr/bin/env bash
# Orchestrate data prep, training ablations, evaluation, visualization, active learning, and summaries.
# Run from repo root: `bash run_pipeline.sh`

set -euo pipefail

export PYTHONPATH=.

echo "=== 1) Data preparation ==="
python notebooks/01_data_preparation.py

echo "=== 2) Training: fusion (baseline) ==="
python -m src.train --device auto --num_epochs 12 --batch_size 32 --mode fusion

echo "=== 3) Training: video_only ablation ==="
python -m src.train --device auto --num_epochs 12 --batch_size 32 --mode video_only

echo "=== 4) Training: rheology_only ablation ==="
python -m src.train --device auto --num_epochs 12 --batch_size 32 --mode rheology_only

echo "=== 5) Evaluation on test set (uses latest best_model.pt) ==="
python -m src.evaluate --checkpoint_path results/models/best_model.pt --device auto

echo "=== 6) Visualization (UMAP + examples) ==="
python notebooks/02_model_training.py

echo "=== 7) Active learning: uncertainty + random baselines ==="
python -m src.active_learning --device auto --n_rounds 5 --epochs_per_round 3 --batch_size 64 --acquire_per_round 50

echo "=== 8) Active learning overlay figure ==="
python notebooks/03_active_learning_demo.py

echo "=== 9) Summaries (tables for README) ==="
python notebooks/04_results_summary.py

echo "All stages completed. Check results/ for metrics, models, and figures."
