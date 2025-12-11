"""Training script for the multi-modal fusion model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import config
from src.datasets import create_dataloaders
from src.fusion_model import MultiModalFusionModel, compute_losses


def _move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _apply_mode(batch: Dict, mode: str) -> Dict:
    """
    Apply modality ablations for training modes.

    mode:
        fusion (default): use all modalities.
        video_only: zero out rheology and mark missing.
        rheology_only: zero out video and mark missing.
    """
    b = batch.copy()
    missing = b.get("missing_modalities", {}) if isinstance(b.get("missing_modalities", {}), dict) else {}
    if mode == "video_only":
        if "rheology" in b:
            b["rheology"] = torch.zeros_like(b["rheology"])
        missing["rheology"] = True
    elif mode == "rheology_only":
        if "video" in b:
            b["video"] = torch.zeros_like(b["video"])
        missing["video"] = True
    b["missing_modalities"] = missing
    return b


def train_model(args: argparse.Namespace) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    mode_suffix = {
        "fusion": "fusion",
        "video_only": "video_only",
        "rheology_only": "rheo_only",
    }.get(args.mode, args.mode)
    metrics_path = Path("results") / f"training_metrics_{mode_suffix}.csv"

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        use_microscopy=args.use_microscopy,
        num_workers=0,
        device="cpu",  # keep CPU in dataset; move to device in loop
    )

    model = MultiModalFusionModel(
        use_microscopy=args.use_microscopy,
        num_classes=config.NUM_CLASSES,
        num_reg_targets=config.NUM_REG_TARGETS,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.1)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience = args.patience
    wait = 0
    history = []
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_batches = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs} [train]")
        for batch in train_iter:
            batch = _move_to_device(batch, device)
            batch = _apply_mode(batch, args.mode)
            optimizer.zero_grad()
            outputs = model(batch)
            total_loss, class_loss, reg_loss = compute_losses(
                batch, outputs, alpha_reg=args.alpha_reg
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += total_loss.item() * batch["video"].size(0)
            preds = outputs["logits_class"].argmax(dim=1)
            correct += (preds == batch["y_class"]).sum().item()
            total += batch["y_class"].size(0)
            train_batches += 1
            train_iter.set_postfix(
                loss=total_loss.item(), class_loss=class_loss.item(), reg_loss=reg_loss.item()
            )

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_mae = 0.0
        val_running_loss = 0.0
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{args.num_epochs} [val]")
            for batch in val_iter:
                batch = _move_to_device(batch, device)
                batch = _apply_mode(batch, args.mode)
                outputs = model(batch)
                v_loss, _, _ = compute_losses(batch, outputs, alpha_reg=args.alpha_reg)
                preds = outputs["logits_class"].argmax(dim=1)
                val_correct += (preds == batch["y_class"]).sum().item()
                val_total += batch["y_class"].size(0)
                val_mae += torch.abs(outputs["preds_reg"] - batch["y_reg"]).mean().item() * batch[
                    "y_reg"
                ].size(0)
                val_running_loss += v_loss.item() * batch["video"].size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_mae = val_mae / val_total if val_total > 0 else 0.0
        val_loss = val_running_loss / val_total if val_total > 0 else 0.0

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, val_mae={val_mae:.4f}"
        )

        # Early stopping and best model saving (by val_acc)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt_path = save_dir / f"best_model_{mode_suffix}.pt"
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc}, ckpt_path)
            # For backward compatibility, also write a generic best_model.pt for fusion mode
            if args.mode == "fusion":
                torch.save({"model_state": model.state_dict(), "val_acc": val_acc}, save_dir / "best_model.pt")
            print(f"Saved best model to {ckpt_path}")
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping triggered.")
            break

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_mae": val_mae,
            }
        )

    # Save training metrics
    import csv

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved training metrics to {metrics_path}")

    # Plot training curves
    try:
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, [h["train_acc"] for h in history], label="train_acc")
        plt.plot(epochs, [h["val_acc"] for h in history], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"training_curves_{mode_suffix}.png")
        plt.close()
        # For fusion mode, also save a generic plot for compatibility
        if args.mode == "fusion":
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, [h["train_acc"] for h in history], label="train_acc")
            plt.plot(epochs, [h["val_acc"] for h in history], label="val_acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "training_curves.png")
            plt.close()
        print(f"Saved training curves to {figures_dir / f'training_curves_{mode_suffix}.png'}")
    except Exception as e:
        print(f"Could not plot training curves: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-modal fusion model")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num_epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--alpha_reg", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_microscopy", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["fusion", "video_only", "rheology_only"],
        help="Training mode for ablations.",
    )
    parser.add_argument("--save_dir", type=str, default="results/models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
