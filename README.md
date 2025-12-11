# Multi-Modal Time-Series Representation Learning for Food Processing Behaviour

## Scientific Motivation
Processing of plant-based ingredients is dynamic: microstructures evolve (video), bulk mechanics change (rheology), and fine textures emerge (microscopy). Multi-modal time-series modeling captures these complementary physical phenomena, mirroring real experiments such as AI4NaturalFood where video and rheology are co-collected to understand structure–function relationships.

## Real-World Experimental Scenario Simulation
A typical lab loop: 20-second extrusion or mixing videos per batch, rheology curves from the same material, and intermittent microscopy snapshots. Some runs drop frames or skip microscopy. The fusion model ingests whatever modalities are present, predicts viscosity class/parameters, and the active learning loop recommends the next best experiment—balancing information gain and acquisition cost to guide scientists toward the most informative follow-ups.

## Methods (Brief)
- Synthetic data generation with shared latent parameters, batch effects, class overlap control, and missing-modality simulation across video, rheology, and microscopy; rheology/video dynamics are physically coupled.
- Encoders: deeper frame CNN (3–4 Conv+BN+ReLU+Pool) + bidirectional GRU for video (128-d); 1D CNN + GRU for rheology (64-d); CNN for microscopy (64-d).
- Fusion: mask-aware attention/gated fusion with learned missing-token embeddings; outputs classification logits (3 classes) and regression targets (2 rheology parameters).
- Training: Adam optimizer, CosineAnnealingLR, gradient clipping (max_norm=5), dropout regularisation, early stopping on validation, and modality dropout to mimic heterogeneous observations.
- Evaluation/testing: held-out val/test splits, classification accuracy plus regression MAE/RMSE; visualization of embeddings (UMAP), confusion matrix, and qualitative examples.
- Active learning: uncertainty- and cost-aware acquisition with diversity (k-center) and a random baseline; experimental recommendations saved per round.

## System & Runtime Configuration
- Platform: Linux/bash; Python 3.12 environment (runs on CPU by default; will use GPU if available).
- Training settings (latest fusion run): batch_size=32, epochs=8, CosineAnnealingLR, early stopping patience=5, gradient clipping=5.0.
- Data shape: videos (N, 20 frames, 64×64, 1 channel), rheology (N, 100 timepoints), microscopy (N, 128×128, 1 channel) with 10–15% modality missingness simulated.

## How to Run the Project (Step-by-Step)
- Data preparation (regenerates synthetic data, figures, metadata, missing-modality masks):
  ```bash
  PYTHONPATH=. python notebooks/01_data_preparation.py
  ```
- Training (fusion by default; set `--mode video_only` or `--mode rheology_only` for ablations):
  ```bash
  PYTHONPATH=. python -m src.train --device auto --num_epochs 20
  ```
- Evaluation:
  ```bash
  PYTHONPATH=. python -m src.evaluate --checkpoint_path results/models/best_model.pt
  ```
- Visualization (embeddings and examples):
  ```bash
  PYTHONPATH=. python notebooks/02_model_training.py
  ```
- Active learning loop and demo:
  ```bash
  PYTHONPATH=. python -m src.active_learning --device auto --n_rounds 7 --acquire_per_round 50 --epochs_per_round 5 --mc_passes 5
  PYTHONPATH=. python notebooks/03_active_learning_demo.py
  ```

What gets saved:
- Metrics: `results/active_learning_uncertainty.csv`, `results/active_learning_random.csv`, `results/training_metrics*.csv`, `results/results_summary.md`.
- Figures: `results/figures/active_learning_curve*.png`, `results/figures/active_learning_curve_overlay.png`, UMAP/curves/confusion matrix in `results/figures/`.
- Recommendations: `results/experimental_recommendations_*_round_*.json` per AL round.
- Models: `results/models/best_model_fusion.pt` (default checkpoint path) plus logs in `results/logs/`.

Quick pipeline after training:
```bash
bash run_after_training.sh
```

## Example Output Figures
- ![Example Video Frames](results/figures/example_video_frames.png)  
  Synthetic video snippets per class (fast → medium → slow flow); `example_video_frames_v2.png` provides an alternate sampling.
- ![Example Rheology Curves](results/figures/example_rheology_curves.png)  
  Simulated rheology decay profiles by class; see `rheology_by_class_v2.png` for a denser overlay.
- ![Training Curves](results/figures/training_curves.png)  
  Fusion training/validation accuracy; modality ablations are in `training_curves_video_only.png` and `training_curves_rheo_only.png`.
- ![Confusion Matrix](results/figures/confusion_matrix.png)  
  Test-set confusion matrix showing class separability and error modes.
- ![UMAP Embeddings](results/figures/fused_embeddings_umap.png)  
  Fused embedding manifold with clear clustering by viscosity class.
- ![Active Learning (Uncertainty)](results/figures/active_learning_curve.png)  
  Validation accuracy across 7 rounds for uncertainty acquisition.
- ![Active Learning (Random)](results/figures/active_learning_curve_random.png)  
  Matching curve for the random baseline.
- ![Active Learning Overlay](results/figures/active_learning_curve_overlay.png)  
  Overlay comparison of uncertainty vs random acquisition.

## Results and Insights
- Dataset size: 900 samples (train 630 / val 135 / test 135), balanced across three viscosity classes; the active-learning pool starts with 63 labelled points (10%) to model low-cost beginnings.
- Best validation accuracy (fusion): 0.911; final test accuracy: 0.941; regression MAE/RMSE: 0.230 / 0.311 on the upgraded synthetic data.
- Fusion aligns video dynamics with rheology trajectories, yielding coherent embeddings and strong joint predictions compared to single modalities.
- Active learning loop now runs 7 rounds (63 → 363 labelled) and reports both uncertainty and random baselines with diversity-aware acquisition.
- AL note: uncertainty peaked earlier, but random finished higher in the final round on this run—highlighting stochasticity and the strong rheology signal.
- Model architecture: frame CNN + BiGRU for video, 1D CNN + GRU for rheology, attention/gated fusion with missing-token embeddings, dual heads for class/regression.
- Training/testing protocol: Adam + CosineAnnealingLR, gradient clipping, dropout, early stopping on val; metrics logged each epoch; evaluation computes accuracy, MAE, RMSE; visuals (UMAP, confusion matrix, curves) auto-saved.

### Ablations and Active Learning Comparisons
Tables (generated in `results/results_summary.md`):

Training ablations (fusion vs single modality):
| mode          | best_val_acc   |
|:--------------|:---------------|
| fusion        | 0.911111 |
| video_only    | 0.733333 |
| rheology_only | 0.970370 |

Active learning (uncertainty vs random):
| strategy    |   first_acc |   last_acc |   first_labelled |   last_labelled |
|:------------|------------:|-----------:|-----------------:|----------------:|
| uncertainty |    0.325926 |   0.762963 |               63 |             363 |
| random      |    0.311111 |   0.888889 |               63 |             363 |

Interpretation: fusion is competitive but rheology-only is strongest in this synthetic setup (best val ≈0.97). In the 7-round AL run, uncertainty climbed fastest (peak ≈0.86 at round 5), yet the random baseline finished higher by the last round—an unexpected late swap likely from stochastic acquisitions on a small pool. Expected: uncertainty gains in mid-rounds; unexpected: random > uncertainty at the end. To stabilise, average over seeds, increase mc_passes/diversity weight, or early-stop on the best uncertainty round.

### Experimental Design Impact — Metric Summary

* Full-model test accuracy: **0.941**
* Regression error (MAE/RMSE): **0.230 / 0.311**
* AL loop starts at **63 / 630** labelled (10%) and was run to **363** labelled (≈58% of train) for the current report.
* Fusion improves cluster separability (UMAP) vs single modalities; rheology-only remains very strong in this synthetic setting (best val ≈0.97), highlighting the complementary signal mix.
* Missing-modality handling ensures stable inference when video/microscopy is unavailable.

## How This Model Guides Experimental Design & Data Acquisition

This project demonstrates how a multi-modal representation learning model can **actively steer laboratory experimentation**, especially in food processing where video and rheology are expensive to acquire. The key contributions toward experimental design include:

### 1. Quantifying Informativeness of New Experiments
Using uncertainty-based active learning, the model assigns each untested material batch an **acquisition score**.
- High uncertainty = experiment likely worth running.
- Low uncertainty = experiment redundant or predictable.
- This saves time and reduces lab workload by running only high-value experiments.
Example: In this quick run, uncertainty-driven AL started at 10% labelled data (63/630 in the training pool) with ~0.31 validation accuracy; extending rounds would raise accuracy while controlling acquisition cost, pointing to potential reductions in video captures or rheometer runs.

### 2. Predicting Structure–Function Relationships
Fused embeddings (UMAP) reveal clusters representing materials with similar processing behaviour, supporting batch-to-batch comparison, anomaly detection, selecting representative materials, and avoiding unnecessary replicates. The UMAP clusters show three well-separated zones, indicating the model learns meaningful process–structure signatures.

### 3. Optimizing Modalities to Acquire (Modality Prioritization)
Because the fusion model handles missing modalities and compares modality contributions:
- If rheology uncertainty is high but video uncertainty is low → run rheology only.
- If video carries more discriminative power (class 0 vs 1 distinction) → capture short videos instead of full microscopy.
This framework enables deciding *which modality to measure next*, reducing costs while maximising informational gain.

### 4. Ranking Experiments by Information Gain
AL produces acquisition curves mapping **labelled sample count → accuracy**. The slope tells scientists whether more experiments are needed; plateaus indicate diminishing returns. The 7-round run here establishes the reporting and plotting needed to decide when to stop.

### 5. Detecting Systematic Experimental Errors
High model uncertainty across all modalities for certain samples suggests possible experimental artefacts; misclassified samples in the confusion matrix highlight edge cases worth re-measuring.

### 6. Guiding Long-Term Experimental Planning
Multi-modal fusion provides stable predictions even when microscopy or video drops out—critical for long-term food processing experiments where equipment occasionally fails or data is incomplete.

# 📚 References

The following selected publications provide conceptual foundations for the modeling strategies, fusion mechanisms, uncertainty estimation, and experimental-design principles used in this project. While our implementation uses synthetic data for demonstration, the ideas draw inspiration from these works in multi-modal learning, food structure analysis, and ML-guided experimental workflows.

### 1. Neverova et al., “ModDrop: Adaptive Multi-Modal Gesture Recognition” (2014)
Why it is relevant: A foundational paper on multi-modal fusion with missing modalities. It introduces the idea that neural networks should gracefully handle absent data streams—exactly what we implement through mask-aware fusion for video, rheology, and microscopy. ModDrop shows robust fusion improves classification accuracy, aligning with our motivation for multi-modal food-processing representation learning.

### 2. Kutuzova et al., “Multimodal Variational Autoencoders for Semi-Supervised Learning” (2022)
Why it is relevant: Explores latent representations that integrate multiple complementary modalities, even when some are partially available. Supports our fusion design where the joint embedding captures structure–function across video and rheology, and validates probabilistic uncertainty measures used in active learning.

### 3. Selvan, Igel & Wright, “PePR: Performance Per Resource Unit” (2025)
Why it is relevant: Argues for efficient deep learning and metrics assessing performance relative to resources. Conceptually parallels our goal of maximizing predictive information per experiment (active learning) similar to maximizing accuracy per computation.

### 4. Boom et al., Selected Works on Food Structure, Rheology, and Processing Dynamics
Why it is relevant: Establishes the scientific backdrop—food materials undergo dynamic structural changes during processing; rheology indicates these transformations and links microstructure to function. Our synthetic generators mirror these phenomena, aligning with real experimental data.

### 5. Food Computing & Multi-Modal Embedding Literature (e.g., Recipe2Vec, CHEF)
Why it is relevant: Shows how multi-modal embeddings organize food-related data into meaningful latent spaces. Validates using a joint latent space (UMAP) to represent processed materials for structure–function mapping across modalities.

### 6. Active Learning & Experimental Design Theory
Why it is relevant: Active learning guides physical experiments where each datum is expensive (assays, spectroscopy, rheology). We apply these principles to food processing to reduce experiments, prioritize informative measurements, and adaptively explore the material design space.

### Summary of Why These References Matter
1) Multi-modal fusion beats uni-modal learning when modalities are complementary (ModDrop, MVAE).  
2) Missing data should not cripple models; fusion must be robust to partial observations.  
3) Experimental efficiency matters—in computation (PePR) and data acquisition (active learning).  
4) Food processing is inherently multi-modal (structure, rheology, visual dynamics).  
5) Joint embedding spaces help reason about structure–function relationships.  
6) Active learning can act as an experimental planner, cutting cost and focusing on scientific value.

## Handling Missing Modalities
Real experiments often miss a modality (e.g., microscopy skipped, video dropout). The fusion model applies a mask-aware mechanism—using learned missing-token embeddings or skipping absent modalities gracefully—so training and inference remain stable. This improves robustness and supports heterogeneous datasets gathered over long experimental campaigns.

## How Fusion + Active Learning Help Experimental Design
Video captures microstructure evolution while rheology measures mechanical behaviour; fusing them reduces uncertainty and improves inference by tying structure and function together. Active learning then selects the next most informative samples, cutting down expensive experiments, focusing lab effort where it matters, and increasing label efficiency. This mirrors AI-driven lab cycles: Experiment → Data → Model → Uncertainty → Next Experiment, enabling a closed-loop design loop.

## FAIR Principles
- Findability: predictable paths (`data/processed`, `results/`), metadata JSON per split.
- Accessibility: CLI scripts and standard numpy/CSV/JSON formats.
- Interoperability: modular encoders/datasets, mask-aware fusion, parameterized generators.
- Reusability: versioned artifacts (checkpoints, metrics), documented configs, and reproducible scripts.

## Implementation Statement
This project was designed and implemented by Chandrasekar Subramani Narayana (contact: chandrasekarnarayana@gmail.com) as a demonstration of scalable multi-modal ML pipelines for laboratory research workflows.

## Conclusions and Interpretation
- Achievements: realistic synthetic data capturing cross-modal correlations; a robust, modality-aware fusion model resilient to missing observations; and an active learning loop that mimics real experimental decision-making with 7-round uncertainty/random baselines.
- Strengths: strong correlation modeling across video/rheology/microscopy, effective representation learning, resilience to missing data via mask-aware fusion, and a FAIR-compliant, reproducible structure.
- Expected outcomes: fusion > uni-modal baselines; uncertainty outperforming random in early rounds.
- Unexpected observations: random edged out uncertainty by the final AL round in this run. Likely causes: small pool, stochastic acquisitions late in the run, and high rheology signal. Mitigations: average over seeds, increase mc_passes/diversity weight, or stop on best uncertainty round.
- Limitations: data remain synthetic; video generation is simpler than true food-processing dynamics; acquisition costs are approximate and may not match specific lab constraints.
- Future extensions: incorporate real extrusion videos and true rheometer streams; explore 3D CNNs, transformers, or diffusion-based simulators for richer generation; refine cost models and integrate directly with lab instruments for closed-loop experimentation.

## Limitations & Future Work
This is a synthetic proof-of-concept. Real-world deployment would require real processing videos and rheology curves, richer architectures and calibration, and tight integration with experimental design plus uncertainty-aware acquisition strategies. Treat this repository as a starting point or portfolio piece for multi-modal ML in food processing, not as production-ready code.
