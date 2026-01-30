"""
Train DistilHuBERT-based Audio Emotion Classifier on CREMA-D
------------------------------------------------------------

This script fine-tunes a DistilHuBERT backbone on CREMA-D using:
- Raw waveforms (16 kHz) as input to DistilHuBERT
- Prosodic features (pitch, energy, speech-rate proxy, etc.) from `audio_features`

The resulting model produces:
- Emotion logits
- A compact audio embedding that can be fused with a text-based model
  (e.g., RoBERTa fine-tuned on GoEmotions [https://www.tensorflow.org/datasets/catalog/goemotions?utm_source=chatgpt.com]).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from typing import Dict

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

import config
from src.audio_emotion_dataset import (
    CREMAAudioEmotionDataset,
    collate_fn,
)
from src.audio_emotion_model import (
    AudioEmotionConfig,
    DistilHuBERTEmotionModel,
)


def build_label_mapping() -> Dict[str, int]:
    """
    Build emotion label mapping based on `config.EMOTION_MAPPING`.

    CREMA-D in this project uses 6 basic emotions:
        anger, disgust, fear, happy, neutral, sad
    """
    # Values of EMOTION_MAPPING are canonical emotion names
    emotions = sorted(set(config.EMOTION_MAPPING.values()))
    label2id = {emo: i for i, emo in enumerate(emotions)}
    return label2id


def train_one_epoch(
    model: DistilHuBERTEmotionModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    from tqdm import tqdm
    
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for batch_idx, batch in enumerate(pbar):
        waveforms = batch["waveforms"].to(device)
        prosodics = batch["prosodics"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(waveforms, prosodics)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * waveforms.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: DistilHuBERTEmotionModel,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in dataloader:
        waveforms = batch["waveforms"].to(device)
        prosodics = batch["prosodics"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(waveforms, prosodics)
        loss = criterion(logits, labels)
        total_loss += loss.item() * waveforms.size(0)

        preds = torch.argmax(logits, dim=-1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {"loss": avg_loss, "accuracy": acc, "macro_f1": macro_f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilHuBERT + prosodic features on CREMA-D"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=str(config.SPLITS_DIR / "train.csv"),
        help="Path to training split CSV",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=str(config.SPLITS_DIR / "val.csv"),
        help="Path to validation split CSV",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.PROCESSED_DATA_DIR / "audio_emotion_model"),
        help="Directory to save fine-tuned model",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze DistilHuBERT backbone and only train classifier head",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Labels and datasets
    label2id = build_label_mapping()
    id2label = {v: k for k, v in label2id.items()}
    print(f"Label mapping: {label2id}")

    train_ds = CREMAAudioEmotionDataset(
        csv_path=Path(args.train_csv),
        label2id=label2id,
    )
    val_ds = CREMAAudioEmotionDataset(
        csv_path=Path(args.val_csv),
        label2id=label2id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Model
    cfg = AudioEmotionConfig(num_labels=len(label2id))
    model = DistilHuBERTEmotionModel(cfg)
    model.to(device)

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Frozen DistilHuBERT backbone; training classifier + prosody fusion only.")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    best_val_f1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, criterion
        )
        metrics = evaluate(model, val_loader, device, criterion)

        print(
            f"  Train loss: {train_loss:.4f} | "
            f"Val loss: {metrics['loss']:.4f} | "
            f"Val acc: {metrics['accuracy']:.4f} | "
            f"Val macro-F1: {metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            ckpt_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "config": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved new best model to {ckpt_path}")

    print("\nTraining complete.")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()


