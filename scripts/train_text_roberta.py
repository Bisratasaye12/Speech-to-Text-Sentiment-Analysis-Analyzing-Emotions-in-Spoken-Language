"""
Train RoBERTa (with LoRA) for 6-way text emotion classification on GoEmotions.

Key design choices:
- Uses stratified train/val split from `data/raw/goemotions_6class_train_full.csv`
- Mode A by default: freeze base RoBERTa, train only LoRA + classifier head
- Optimizer: AdamW with standard CrossEntropyLoss
- Metrics: track accuracy AND macro-F1 (critical for class imbalance)
- Early stopping based on validation macro-F1 (not accuracy)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging

# Suppress expected warnings
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Suppress transformers logging warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Ensure project root is on sys.path so we can import `config` and `src.*`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.text_dataset import create_datasets_from_csv
from src.text_model import load_roberta_text_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Use MPS (Metal Performance Shaders) for Apple GPU acceleration
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": float(acc), "macro_f1": float(macro_f1)}


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    show_progress: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False, disable=not show_progress)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    show_progress: bool = True,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Validating", leave=False, disable=not show_progress)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"].detach().cpu().numpy()
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits.detach().cpu().numpy()

        total_loss += loss.item()
        all_logits.append(logits)
        all_labels.append(labels)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(1, len(dataloader))
    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(logits_arr, labels_arr)
    return avg_loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RoBERTa+LoRA for 6-way text emotion classification."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(config.RAW_DATA_DIR / "goemotions_6class_train_full.csv"),
        help="Path to the GoEmotions 6-class CSV (text,label).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (L2 regularization).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps used for warmup.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping patience based on val macro-F1.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max sequence length for tokenizer.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints. Default: data/processed/models/checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1, saves after each epoch).",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Setup checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = config.PROCESSED_DATA_DIR / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load model + tokenizer (Mode A by default: LoRA + frozen base)
    artifacts = load_roberta_text_model(
        model_name="roberta-base",
        use_lora=True,
        freeze_base=True,
    )
    model = artifacts.model.to(device)
    tokenizer = artifacts.tokenizer

    # Datasets and loaders with stratified split
    csv_path = Path(args.csv_path)
    train_ds, val_ds, label_enc = create_datasets_from_csv(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_ratio=0.1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Optimizer: only trainable parameters (LoRA + head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Resume from checkpoint if provided
    start_epoch = 1
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_macro_f1 = checkpoint.get("best_macro_f1", -1.0)
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        print(f"Resumed from epoch {checkpoint['epoch']}, best macro-F1: {best_macro_f1:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, show_progress=True)
        val_loss, val_metrics = evaluate(model, val_loader, device, show_progress=True)

        print(
            f"  train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        # Early stopping based on macro-F1 (especially important with imbalance)
        is_best = val_metrics["macro_f1"] > best_macro_f1
        if is_best:
            best_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            print(f"  ✓ New best macro-F1: {best_macro_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{args.patience})")

        # Save checkpoint after each epoch (or every N epochs)
        if epoch % args.save_every == 0 or is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_macro_f1": best_macro_f1,
                "epochs_without_improvement": epochs_without_improvement,
                "label2id": label_enc.label2id,
                "id2label": label_enc.id2label,
                "val_metrics": val_metrics,
            }
            # Save regular checkpoint
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, ckpt_path)
            # Save best checkpoint separately
            if is_best:
                best_ckpt_path = checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_ckpt_path)
                print(f"  Saved best checkpoint: {best_ckpt_path.name}")

        if epochs_without_improvement >= args.patience:
            print("\nEarly stopping triggered based on macro-F1.")
            break

    # Final summary
    models_dir = config.PROCESSED_DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    final_best_path = models_dir / "text_roberta_lora_best.pt"
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    if best_checkpoint_path.exists():
        shutil.copy(best_checkpoint_path, final_best_path)
        print(f"\n✓ Training complete. Best model: {final_best_path.relative_to(config.PROJECT_ROOT)}")
        print(f"  Best macro-F1: {best_macro_f1:.4f}")
    else:
        print("\nTraining completed but no best checkpoint found.")


if __name__ == "__main__":
    main()


