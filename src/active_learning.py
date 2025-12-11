"""Pool-based active learning utilities and experiment runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import config
from src.datasets import MultiModalTimeSeriesDataset
from src.fusion_model import MultiModalFusionModel, compute_losses


def _log(message: str) -> None:
    """Print with flushing to keep logs streaming in long runs."""
    print(message, flush=True)


@dataclass
class ActiveLearningPool:
    """Tracks labelled and unlabelled indices for pool-based active learning."""

    num_samples: int
    initial_fraction: float = 0.1
    random_state: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.random_state)
        n_initial = max(1, int(self.initial_fraction * self.num_samples))
        all_indices = np.arange(self.num_samples)
        init_indices = rng.choice(all_indices, size=n_initial, replace=False)
        self.indices_labelled = set(init_indices.tolist())
        self.indices_unlabelled = set(all_indices.tolist()) - self.indices_labelled

    def get_labelled_indices(self) -> List[int]:
        return sorted(self.indices_labelled)

    def get_unlabelled_indices(self) -> List[int]:
        return sorted(self.indices_unlabelled)

    def add_labelled(self, new_indices: Sequence[int]) -> None:
        for idx in new_indices:
            if idx in self.indices_unlabelled:
                self.indices_unlabelled.remove(idx)
                self.indices_labelled.add(idx)


def _predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-8)).sum(dim=1)


def select_random_samples(pool_indices: Sequence[int], k: int, rng: np.random.Generator) -> List[int]:
    """Select k samples uniformly at random from pool."""
    if len(pool_indices) == 0:
        return []
    k = min(k, len(pool_indices))
    return rng.choice(pool_indices, size=k, replace=False).tolist()


def select_most_uncertain(
    model: MultiModalFusionModel,
    dataset: MultiModalTimeSeriesDataset,
    pool_indices: Sequence[int],
    k: int,
    device: torch.device,
    batch_size: int = 32,
    mc_passes: int = 2,
    use_diversity: bool = True,
) -> List[int]:
    """
    Select top-k samples maximizing information per cost with multimodal uncertainty and diversity.
    """
    model.train()  # enable dropout for MC estimates
    subset = Subset(dataset, pool_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    video_cost = 1.0
    rheo_cost = 0.5
    micro_cost = 0.7

    scores = []
    indices = []
    embeddings = []

    softmax = nn.Softmax(dim=1)

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits_mc = []
        reg_mc = []
        embed_fused_list = []
        with torch.no_grad():
            for _ in range(mc_passes):
                outputs = model(batch)
                logits_mc.append(outputs["logits_class"])
                reg_mc.append(outputs["preds_reg"])
                embed_fused_list.append(outputs["embed_fused"])
        logits_mc = torch.stack(logits_mc, dim=0)  # (M, B, C)
        reg_mc = torch.stack(reg_mc, dim=0)  # (M, B, R)

        # Classification entropy using mean probs
        mean_probs = softmax(logits_mc).mean(dim=0)
        entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=1)

        # Regression variance (mean over targets)
        reg_var = reg_mc.var(dim=0).mean(dim=1)

        # Modality-specific uncertainty (proxy via weights if present)
        modality_uncertainty = torch.zeros_like(entropy)
        if "modality_weights" in outputs:
            mw = outputs["modality_weights"]
            # High gate variance -> more uncertainty
            modality_uncertainty += mw["video"].var(dim=1) if "video" in mw else 0
            modality_uncertainty += mw["rheology"].var(dim=1) if "rheology" in mw else 0
            if "microscopy" in mw:
                modality_uncertainty += mw["microscopy"].var(dim=1)

        # Combined score
        combined = entropy + reg_var + 0.5 * modality_uncertainty

        # Cost-aware: information per cost
        cost = torch.full_like(combined, video_cost + rheo_cost)
        if "microscopy" in batch and torch.is_tensor(batch["microscopy"]):
            cost += micro_cost
        info_per_cost = combined / cost

        scores.append(info_per_cost.cpu())
        indices.extend(batch["index"].tolist())
        embeddings.append(torch.stack(embed_fused_list, dim=0).mean(dim=0).cpu())
        if (batch_idx + 1) % max(1, len(loader) // 5) == 0 or batch_idx == len(loader) - 1:
            _log(f"[AL]  Scoring pool: {batch_idx+1}/{len(loader)} batches done")

    scores = torch.cat(scores)
    embeddings = torch.cat(embeddings) if embeddings else torch.empty(0)

    # Diversity via k-center greedy on fused embeddings (optional)
    selected = []
    if len(scores) == 0:
        return selected
    remaining = list(range(len(scores)))
    if embeddings.numel() == 0 or not use_diversity:
        topk = torch.topk(scores, k=min(k, len(scores))).indices.numpy()
        return [indices[i] for i in topk]

    # Start with highest score
    first = int(torch.argmax(scores))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < min(k, len(scores)):
        sel_emb = embeddings[selected]
        rem_emb = embeddings[remaining]
        # distance to nearest selected
        dists = torch.cdist(rem_emb, sel_emb).min(dim=1).values
        # combine distance with score
        combined_score = scores[remaining] + 0.1 * dists
        next_idx = int(torch.argmax(combined_score))
        selected.append(remaining[next_idx])
        remaining.pop(next_idx)

    return [indices[i] for i in selected]


def _make_loader(
    dataset: MultiModalTimeSeriesDataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def _train_one_round(
    model: MultiModalFusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    num_epochs: int,
    alpha_reg: float,
) -> tuple[float, float]:
    optimizer = Adam(model.parameters(), lr=lr)
    _log(f"[AL] Training for {num_epochs} epochs on {len(train_loader.dataset)} labelled samples")
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch)
            total_loss, _, _ = compute_losses(batch, outputs, alpha_reg=alpha_reg)
            total_loss.backward()
            optimizer.step()
        if (epoch + 1) % max(1, num_epochs // 2) == 0 or epoch == num_epochs - 1:
            _log(f"[AL]  Epoch {epoch+1}/{num_epochs} done")

    # Validation metrics
    model.eval()
    correct = 0
    total = 0
    mae_sum = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch)
            preds = outputs["logits_class"].argmax(dim=1)
            correct += (preds == batch["y_class"]).sum().item()
            total += batch["y_class"].size(0)
            mae_sum += torch.abs(outputs["preds_reg"] - batch["y_reg"]).mean().item() * batch[
                "y_reg"
            ].size(0)

    val_acc = correct / total if total > 0 else 0.0
    val_mae = mae_sum / total if total > 0 else 0.0
    return val_acc, val_mae


def run_active_learning_experiment(
    n_rounds: int = 5,
    initial_fraction: float = 0.1,
    acquire_per_round: int = 30,
    use_microscopy: bool = False,
    data_dir: str = "data/processed",
    device: str = "cpu",
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs_per_round: int = 5,
    alpha_reg: float = 1.0,
    strategy: str = "uncertainty",
    metrics_path: Path | None = None,
    mc_passes: int = 2,
    use_diversity: bool = True,
) -> None:
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)
    _log(f"[AL] Using device: {device_t}")

    results_dir = Path("results")
    figures_dir = Path("results/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MultiModalTimeSeriesDataset(
        split="train", data_dir=data_dir, use_microscopy=use_microscopy, device="cpu"
    )
    _log(f"[AL] Loaded train dataset with {len(train_dataset)} samples")
    val_dataset = MultiModalTimeSeriesDataset(
        split="val", data_dir=data_dir, use_microscopy=use_microscopy, device="cpu"
    )
    _log(f"[AL] Loaded val dataset with {len(val_dataset)} samples")
    
    _log(f"[AL] Initializing active learning pool with initial fraction={initial_fraction}")
    pool = ActiveLearningPool(num_samples=len(train_dataset), initial_fraction=initial_fraction)
    _log(f"[AL] Initialized pool with {len(pool.get_labelled_indices())} labelled samples")

    # Keep a single model that accumulates knowledge across rounds (re-initialising each round stalls accuracy).
    model = MultiModalFusionModel(
        use_microscopy=use_microscopy,
        num_classes=config.NUM_CLASSES,
        num_reg_targets=config.NUM_REG_TARGETS,
    ).to(device_t)

    metrics = []
    rng = np.random.default_rng(0)
    _log(f"[AL] Starting experiment strategy={strategy}, rounds={n_rounds}, initial_fraction={initial_fraction}, acquire_per_round={acquire_per_round}")
    for round_idx in range(n_rounds):
        _log(f"[AL] Round {round_idx} --------------------------------------")
        labelled_indices = pool.get_labelled_indices()
        unlabelled_indices = pool.get_unlabelled_indices()
        _log(f"[AL] Round {round_idx}: labelled={len(labelled_indices)}, unlabelled={len(unlabelled_indices)}")

        train_loader = _make_loader(train_dataset, labelled_indices, batch_size, shuffle=True)
        val_loader = _make_loader(val_dataset, list(range(len(val_dataset))), batch_size, False)

        val_acc, val_mae = _train_one_round(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device_t,
            lr=lr,
            num_epochs=epochs_per_round,
            alpha_reg=alpha_reg,
        )
        _log(f"[AL] Round {round_idx} metrics: val_acc={val_acc:.3f}, val_mae={val_mae:.4f}")

        metrics.append(
            {
                "round": round_idx,
                "num_labelled": len(labelled_indices),
                "val_accuracy": val_acc,
                "val_reg_MAE": val_mae,
            }
        )
        _log(
            f"Round {round_idx}: labelled={len(labelled_indices)}, "
            f"val_acc={val_acc:.3f}, val_mae={val_mae:.4f}"
        )

        if round_idx < n_rounds - 1 and len(unlabelled_indices) > 0:
            k = min(acquire_per_round, len(unlabelled_indices))
            if strategy == "uncertainty":
                _log(f"[AL] Selecting most uncertain k={k}")
                new_indices = select_most_uncertain(
                    model=model,
                    dataset=train_dataset,
                    pool_indices=unlabelled_indices,
                    k=k,
                    device=device_t,
                    batch_size=batch_size,
                    mc_passes=mc_passes,
                    use_diversity=use_diversity,
                )
            else:
                _log(f"[AL] Selecting random k={k}")
                new_indices = select_random_samples(unlabelled_indices, k=k, rng=rng)
            _log(f"[AL] Acquired indices (first 10 shown): {new_indices[:10]}")
            pool.add_labelled(new_indices)
            # Save experimental recommendations
            recs = []
            for idx in new_indices:
                recs.append(
                    {
                        "sample_index": int(idx),
                        "recommendation": "Acquire modalities with highest expected information per cost",
                        "modality_importance": {
                            "video": 1.0,
                            "rheology": 0.7,
                            "microscopy": 0.8 if use_microscopy else 0.0,
                        },
                    }
                )
            rec_path = figures_dir.parent / f"experimental_recommendations_{strategy}_round_{round_idx}.json"
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            with rec_path.open("w") as f:
                json.dump(recs, f, indent=2)
            _log(f"Saved experimental recommendations to {rec_path}")

    # Save metrics
    df = pd.DataFrame(metrics)
    csv_path = metrics_path or (results_dir / f"active_learning_{strategy}.csv")
    df.to_csv(csv_path, index=False)
    _log(f"[AL] Saved metrics to {csv_path}")

    # Plot accuracy curve
    plt.figure(figsize=(7, 5))
    plt.plot(df["num_labelled"], df["val_accuracy"], marker="o")
    plt.xlabel("Number of labelled samples")
    plt.ylabel("Validation accuracy")
    plt.title("Active Learning Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    fig_name = "active_learning_curve.png" if strategy == "uncertainty" else "active_learning_curve_random.png"
    fig_path = figures_dir / fig_name
    plt.savefig(fig_path)
    plt.close()
    _log(f"[AL] Saved active learning curve to {fig_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pool-based active learning experiment.")
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--initial_fraction", type=float, default=0.1)
    parser.add_argument("--acquire_per_round", type=int, default=50)
    parser.add_argument("--use_microscopy", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or auto")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--epochs_per_round", type=int, default=5)
    parser.add_argument("--alpha_reg", type=float, default=1.0)
    parser.add_argument("--strategy", type=str, default="uncertainty", choices=["uncertainty", "random"])
    parser.add_argument("--mc_passes", type=int, default=2, help="Monte Carlo dropout passes for uncertainty")
    parser.add_argument("--no_diversity", action="store_true", help="Disable diversity (k-center) in acquisition")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log(f"[AL] main starting with args: {args}")
    # Uncertainty-based AL
    _log("[AL] Running uncertainty-based active learning experiment")
    run_active_learning_experiment(
        n_rounds=args.n_rounds,
        initial_fraction=args.initial_fraction,
        acquire_per_round=args.acquire_per_round,
        use_microscopy=args.use_microscopy,
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs_per_round=args.epochs_per_round,
        alpha_reg=args.alpha_reg,
        strategy="uncertainty",
        metrics_path=Path("results/active_learning_uncertainty.csv"),
        mc_passes=args.mc_passes,
        use_diversity=not args.no_diversity,
    )
    # Random baseline AL
    _log("[AL] Running random baseline active learning experiment")
    run_active_learning_experiment(
        n_rounds=args.n_rounds,
        initial_fraction=args.initial_fraction,
        acquire_per_round=args.acquire_per_round,
        use_microscopy=args.use_microscopy,
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs_per_round=args.epochs_per_round,
        alpha_reg=args.alpha_reg,
        strategy="random",
        metrics_path=Path("results/active_learning_random.csv"),
        mc_passes=args.mc_passes,
        use_diversity=not args.no_diversity,
    )
    _log("[AL] main completed")


if __name__ == "__main__":
    main()
