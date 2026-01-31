"""
Improved Training Script for Audio Emotion Recognition
-----------------------------------------------------

Features:
- Data augmentation (time shift, pitch shift, noise, speed change)
- Class-weighted loss for handling imbalanced classes
- Learning rate scheduling with warmup
- Early stopping
- Gradient clipping
- Better hyperparameters
- More training epochs
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from typing import Dict
from collections import Counter

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


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
from src.audio_augmentation import AudioAugmentation


class AugmentedCREMAAudioEmotionDataset(CREMAAudioEmotionDataset):
    """Dataset with augmentation support - works with both CREMA-D and multi-dataset"""
    
    def __init__(self, *args, augment: bool = False, aug_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        self.aug_config = aug_config or AudioAugmentation()
        # Import here to avoid circular imports
        from src.audio_features import extract_prosodic_features, prosodic_feature_vector
        self.extract_prosodic_features = extract_prosodic_features
        self.prosodic_feature_vector = prosodic_feature_vector
        
        # Update paths for multi-dataset if needed
        if hasattr(self, 'df') and 'dataset' in self.df.columns:
            # Re-resolve paths for multi-dataset
            self.paths = []
            for idx, row in self.df.iterrows():
                p = row[self.audio_col]
                path = Path(p)
                if not path.is_absolute():
                    dataset = row.get("dataset", "CREMA-D")
                    if dataset == "CREMA-D":
                        path = config.PROCESSED_DATA_DIR / path
                    elif dataset == "TESS":
                        if (config.AUDIO_DIR / f"TESS_{path.name}").exists():
                            path = config.AUDIO_DIR / f"TESS_{path.name}"
                        elif (config.RAW_DATA_DIR / "TESS" / path.name).exists():
                            path = config.RAW_DATA_DIR / "TESS" / path.name
                        else:
                            path = config.PROCESSED_DATA_DIR / path
                    elif dataset == "IEMOCAP":
                        if (config.AUDIO_DIR / f"IEMOCAP_{path.name}").exists():
                            path = config.AUDIO_DIR / f"IEMOCAP_{path.name}"
                        else:
                            path = config.PROCESSED_DATA_DIR / path
                    else:
                        path = config.PROCESSED_DATA_DIR / path
                self.paths.append(path)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        label = self.labels[idx]

        audio = self._load_audio(path)
        
        # Apply augmentation during training
        if self.augment:
            audio = self.aug_config.apply(audio, sr=self.sr)
        
        prosodic_dict = self.extract_prosodic_features(audio, sr=self.sr)
        prosodic_vec = self.prosodic_feature_vector(prosodic_dict)

        waveform_tensor = torch.from_numpy(audio)
        prosodic_tensor = torch.from_numpy(prosodic_vec)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "waveform": waveform_tensor,
            "prosodic": prosodic_tensor,
            "label": label_tensor,
        }


def build_label_mapping() -> Dict[str, int]:
    """Build emotion label mapping"""
    emotions = sorted(set(config.EMOTION_MAPPING.values()))
    label2id = {emo: i for i, emo in enumerate(emotions)}
    return label2id


def compute_class_weights(dataset: CREMAAudioEmotionDataset) -> torch.Tensor:
    """Compute class weights for balanced training"""
    labels = np.array(dataset.labels)
    unique_labels = np.array(sorted(set(labels)))
    
    # Compute class weights (inverse frequency)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )
    
    # Convert to tensor
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    return weight_tensor


def create_weighted_sampler(dataset: CREMAAudioEmotionDataset) -> WeightedRandomSampler:
    """Create weighted sampler for balanced sampling"""
    labels = np.array(dataset.labels)
    unique_labels = np.array(sorted(set(labels)))
    
    # Compute sample weights
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    sample_weights = [class_weights[np.where(unique_labels == label)[0][0]] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def train_one_epoch(
    model: DistilHuBERTEmotionModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    criterion: nn.Module,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
) -> float:
    from tqdm import tqdm
    
    model.train()
    total_loss = 0.0
    
    # Use autocast for mixed precision training
    # MPS uses float16, CUDA uses bfloat16, CPU uses float32
    autocast_dtype = torch.float16 if device.type == 'mps' else (torch.bfloat16 if device.type == 'cuda' else torch.float32)
    # MPS doesn't need GradScaler, but CUDA does
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None

    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for batch_idx, batch in enumerate(pbar):
        # Ensure float32 for MPS compatibility before moving to device
        waveforms = batch["waveforms"].float().to(device)
        prosodics = batch["prosodics"].float().to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                logits, _ = model(waveforms, prosodics)
                loss = criterion(logits, labels)
            
            if scaler:  # CUDA with scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:  # MPS or CPU - no scaler needed
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        else:
            logits, _ = model(waveforms, prosodics)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * waveforms.size(0)
        
        # Clear MPS cache periodically to prevent memory buildup
        if device.type == 'mps' and (batch_idx + 1) % 10 == 0:
            torch.mps.empty_cache()
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{current_lr:.2e}"
        })

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: DistilHuBERTEmotionModel,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    
    autocast_dtype = torch.float16 if device.type == 'mps' else torch.bfloat16 if device.type == 'cuda' else torch.float32

    for batch in dataloader:
        # Ensure float32 for MPS compatibility before moving to device
        waveforms = batch["waveforms"].float().to(device)
        prosodics = batch["prosodics"].float().to(device)
        labels = batch["labels"].to(device)

        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                logits, _ = model(waveforms, prosodics)
                loss = criterion(logits, labels)
        else:
            logits, _ = model(waveforms, prosodics)
            loss = criterion(logits, labels)
        total_loss += loss.item() * waveforms.size(0)

        preds = torch.argmax(logits, dim=-1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improved training for audio emotion recognition"
    )
    # Check for multi-dataset splits first, then fallback to CREMA-D
    multi_dataset_train = config.SPLITS_DIR / "multi_dataset" / "train.csv"
    multi_dataset_val = config.SPLITS_DIR / "multi_dataset" / "val.csv"
    default_train = str(multi_dataset_train if multi_dataset_train.exists() else config.SPLITS_DIR / "train.csv")
    default_val = str(multi_dataset_val if multi_dataset_val.exists() else config.SPLITS_DIR / "val.csv")
    
    parser.add_argument(
        "--train_csv",
        type=str,
        default=default_train,
        help="Path to training split CSV",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=default_val,
        help="Path to validation split CSV",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,  # Reduced back to 16 - larger batches causing slowdown on MPS
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=3,
        help="Number of warmup epochs for learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.PROCESSED_DATA_DIR / "audio_emotion_model_improved"),
        help="Directory to save model",
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=True,
        help="Enable data augmentation",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision for faster training",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        default=False,
        help="Compile model with torch.compile (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        default=True,
        help="Use class-weighted loss",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        default=True,
        help="Use weighted random sampler",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone and only train classifier",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="",
        help="Path to a pretrained checkpoint to initialize weights from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Training configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Data augmentation: {args.use_augmentation}")
    print(f"  Class-weighted loss: {args.use_class_weights}")
    print(f"  Weighted sampler: {args.use_weighted_sampler}")
    print(f"  Mixed precision (AMP): {args.use_amp}")
    print(f"  Model compilation: {args.compile_model}")

    # Labels and datasets
    label2id = build_label_mapping()
    id2label = {v: k for k, v in label2id.items()}
    print(f"\nLabel mapping: {label2id}")

    # Augmentation config
    aug_config = AudioAugmentation() if args.use_augmentation else None

    train_ds = AugmentedCREMAAudioEmotionDataset(
        csv_path=Path(args.train_csv),
        label2id=label2id,
        augment=args.use_augmentation,
        aug_config=aug_config,
    )
    val_ds = CREMAAudioEmotionDataset(
        csv_path=Path(args.val_csv),
        label2id=label2id,
    )

    # Print class distribution
    train_labels = train_ds.labels
    label_counts = Counter(train_labels)
    print(f"\nTraining set class distribution:")
    for label_id, count in sorted(label_counts.items()):
        print(f"  {id2label[label_id]:12s}: {count:4d} samples")

    # Create weighted sampler if enabled
    sampler = None
    if args.use_weighted_sampler:
        sampler = create_weighted_sampler(train_ds)
        print("\nUsing weighted random sampler for balanced training")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with augmentation
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,  # MPS doesn't support pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,  # MPS doesn't support pin_memory
    )

    # Model
    cfg = AudioEmotionConfig(
        num_labels=len(label2id),
        hidden_dim=512,  # Larger hidden dimension
        dropout=0.3,  # Slightly higher dropout
    )
    model = DistilHuBERTEmotionModel(cfg)
    model.to(device)
    
    # Optional: initialize from a pretrained checkpoint (e.g., CREMA-D model)
    if args.init_checkpoint:
        ckpt_path = Path(args.init_checkpoint)
        if ckpt_path.exists():
            print(f"\nLoading pretrained weights from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  ⚠ Missing keys when loading checkpoint ({len(missing)}):")
                print(f"    {missing[:10]}{' ...' if len(missing) > 10 else ''}")
            if unexpected:
                print(f"  ⚠ Unexpected keys in checkpoint ({len(unexpected)}):")
                print(f"    {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
            print("  ✓ Pretrained weights loaded")
        else:
            print(f"\n⚠ init_checkpoint not found at: {ckpt_path} — continuing from scratch")
    
    # Compile model for faster training (PyTorch 2.0+)
    if args.compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("\n✓ Model compiled with torch.compile for faster training")
        except Exception as e:
            print(f"\n⚠ Model compilation failed: {e}. Continuing without compilation.")

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("\nFrozen backbone; training classifier only")

    # Class-weighted loss
    if args.use_class_weights:
        class_weights = compute_class_weights(train_ds).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\nUsing class-weighted loss:")
        for label_id, weight in enumerate(class_weights):
            print(f"  {id2label[label_id]:12s}: {weight:.4f}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine annealing after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop with early stopping
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion,
            use_amp=args.use_amp
        )
        metrics = evaluate(model, val_loader, device, criterion, use_amp=args.use_amp)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  Train loss: {train_loss:.4f} | "
            f"Val loss: {metrics['loss']:.4f} | "
            f"Val acc: {metrics['accuracy']:.4f} | "
            f"Val macro-F1: {metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model based on accuracy
        if metrics["accuracy"] > best_val_acc:
            best_val_acc = metrics["accuracy"]
            best_val_f1 = metrics["macro_f1"]
            patience_counter = 0
            
            ckpt_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "val_accuracy": metrics["accuracy"],
                    "val_macro_f1": metrics["macro_f1"],
                },
                ckpt_path,
            )
            print(f"  ✓ Saved new best model (acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                print(f"Best validation macro-F1: {best_val_f1:.4f}")
                break

    print("\n" + "=" * 70)
    print("Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()

