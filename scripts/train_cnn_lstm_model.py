"""
Train CNN+LSTM Emotion Model
-----------------------------

This script trains a CNN+LSTM model for emotion recognition using comprehensive
Librosa features (MFCCs, chroma, spectral, prosodic).

This classic approach often outperforms transformer-based models for emotion recognition.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import librosa

import config
from config import (
    PROCESSED_DATA_DIR,
    CANONICAL_EMOTIONS,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    SPLITS_DIR,
    CNNLSTM_CONFIG,
)
from src.cnn_lstm_emotion_model import (
    CNNLSTMConfig,
    CNNLSTMEmotionModel,
    extract_comprehensive_features,
)


class EmotionDataset(Dataset):
    """Dataset for emotion recognition using CNN+LSTM features"""
    
    def __init__(
        self,
        csv_path: Path,
        label2id: Dict[str, int],
        sr: int = 16000,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.label2id = label2id
        self.sr = sr
        
        # Determine audio path column
        audio_col_candidates = ("processed_filepath", "filepath")
        for col in audio_col_candidates:
            if col in self.df.columns:
                self.audio_col = col
                break
        else:
            raise ValueError(
                f"None of audio columns {audio_col_candidates} found in {csv_path}"
            )
        
        if "emotion" not in self.df.columns:
            raise ValueError(f"'emotion' column not found in {csv_path}")
        
        # Pre-resolve absolute paths (support multi-dataset)
        self.paths: List[Path] = []
        for idx, row in self.df.iterrows():
            p = row[self.audio_col]
            path = Path(p)
            
            if not path.is_absolute():
                # Check if this is a multi-dataset CSV
                dataset = row.get("dataset", "CREMA-D") if "dataset" in row else "CREMA-D"
                if dataset == "CREMA-D":
                    path = PROCESSED_DATA_DIR / path
                elif dataset == "TESS":
                    # Try processed first, then raw
                    audio_dir = config.AUDIO_DIR
                    if (audio_dir / f"TESS_{path.name}").exists():
                        path = audio_dir / f"TESS_{path.name}"
                    elif (config.RAW_DATA_DIR / "TESS" / path.name).exists():
                        path = config.RAW_DATA_DIR / "TESS" / path.name
                    else:
                        path = PROCESSED_DATA_DIR / path
                elif dataset == "IEMOCAP":
                    # Try processed first
                    audio_dir = config.AUDIO_DIR
                    if (audio_dir / f"IEMOCAP_{path.name}").exists():
                        path = audio_dir / f"IEMOCAP_{path.name}"
                    else:
                        path = PROCESSED_DATA_DIR / path
                else:
                    path = PROCESSED_DATA_DIR / path
            
            self.paths.append(path)
        
        # Pre-compute label ids
        self.labels: List[int] = []
        for emo in self.df["emotion"].tolist():
            if emo not in self.label2id:
                raise ValueError(
                    f"Unknown emotion '{emo}' in {csv_path}. "
                    f"Expected one of: {list(self.label2id.keys())}"
                )
            self.labels.append(self.label2id[emo])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        audio_path = self.paths[idx]
        label_id = self.labels[idx]
        
        # Load audio
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio, _ = librosa.load(
            audio_path,
            sr=self.sr,
            mono=config.AUDIO_CONFIG["mono"],
        )
        audio = audio.astype(np.float32)
        
        # Extract features
        features = extract_comprehensive_features(audio, self.sr)
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        return features_tensor, label_id


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function to handle variable-length sequences"""
    features_list, labels_list = zip(*batch)
    
    # Pad sequences to same length
    max_len = max(f.shape[0] for f in features_list)
    feature_dim = features_list[0].shape[1]
    
    padded_features = []
    for f in features_list:
        if f.shape[0] < max_len:
            padding = torch.zeros(max_len - f.shape[0], feature_dim, dtype=f.dtype)
            f = torch.cat([f, padding], dim=0)
        padded_features.append(f)
    
    features = torch.stack(padded_features)
    labels = torch.tensor(labels_list, dtype=torch.long)
    
    return features, labels


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Clear MPS cache before training
    if device.type == "mps":
        torch.mps.empty_cache()
    
    for batch_idx, (features, labels) in enumerate(tqdm(dataloader, desc="Training")):
        # For MPS, process smaller sub-batches if needed
        if device.type == "mps" and features.shape[0] > 4:
            # Split large batches into smaller chunks
            chunk_size = 4
            batch_loss = 0.0
            batch_preds = []
            batch_labels = []
            
            for chunk_start in range(0, features.shape[0], chunk_size):
                chunk_end = min(chunk_start + chunk_size, features.shape[0])
                chunk_features = features[chunk_start:chunk_end].to(device)
                chunk_labels = labels[chunk_start:chunk_end].to(device)
                
                optimizer.zero_grad()
                logits, _ = model(chunk_features)
                loss = criterion(logits, chunk_labels)
                
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_preds.extend(preds)
                batch_labels.extend(chunk_labels.cpu().numpy())
                
                # Clear immediately
                del logits, loss, chunk_features, chunk_labels
                torch.mps.empty_cache()
            
            total_loss += batch_loss / (features.shape[0] / chunk_size)
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
        else:
            # Normal processing for small batches or non-MPS
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            logits, _ = model(features)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Move tensors off GPU immediately
            del logits, loss, features, labels
        
        # Aggressive cache clearing every batch for MPS
        if device.type == "mps":
            torch.mps.empty_cache()
            if (batch_idx + 1) % 3 == 0:  # Every 3 batches
                import gc
                gc.collect()
                torch.mps.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model with memory-efficient processing"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Clear MPS cache before validation
    if device.type == "mps":
        torch.mps.empty_cache()
        import gc
        gc.collect()
    
    with torch.no_grad():
        # Process validation in very small batches or one at a time for MPS
        if device.type == "mps":
            # Process one sample at a time to minimize memory
            for features, labels in tqdm(dataloader, desc="Validating"):
                # Process each sample in batch individually
                batch_size = features.shape[0]
                for i in range(batch_size):
                    single_feature = features[i:i+1].to(device)
                    single_label = labels[i:i+1].to(device)
                    
                    logits, _ = model(single_feature)
                    loss = criterion(logits, single_label)
                    
                    total_loss += loss.item()
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(single_label.cpu().numpy())
                    
                    # Clear immediately
                    del logits, loss, single_feature, single_label
                    torch.mps.empty_cache()
        else:
            # Normal processing for CUDA/CPU
            for features, labels in tqdm(dataloader, desc="Validating"):
                features = features.to(device)
                labels = labels.to(device)
                
                logits, _ = model(features)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                # Move tensors off GPU immediately
                del logits, loss, features, labels
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CNN+LSTM emotion model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "cnn_lstm_model",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (for transfer learning)",
    )
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Device - Use CPU for training to avoid MPS memory issues
    # MPS has memory management issues with bidirectional LSTM
    # CPU is slower but more stable for this model
    use_cpu = os.environ.get("FORCE_CPU", "false").lower() == "true"
    
    if use_cpu:
        device = torch.device("cpu")
        print("Using device: cpu (forced via FORCE_CPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Check if we should use CPU instead due to memory issues
        # MPS has known issues with bidirectional LSTM and memory accumulation
        print("⚠ MPS available but has memory issues with bidirectional LSTM")
        print("   Switching to CPU for stability (use FORCE_CPU=true to force CPU)")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create label mappings
    label2id = {label: i for i, label in enumerate(CANONICAL_EMOTIONS)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Load data from split CSVs (check multi-dataset first, then fallback to CREMA-D)
    print("Loading data from split CSVs...")
    multi_dataset_dir = SPLITS_DIR / "multi_dataset"
    if multi_dataset_dir.exists() and (multi_dataset_dir / "train.csv").exists():
        print("  Using multi-dataset splits")
        train_csv = multi_dataset_dir / "train.csv"
        val_csv = multi_dataset_dir / "val.csv"
        test_csv = multi_dataset_dir / "test.csv"
    else:
        print("  Using CREMA-D splits (multi-dataset not found)")
        train_csv = SPLITS_DIR / "crema_d" / "train.csv"
        val_csv = SPLITS_DIR / "crema_d" / "val.csv"
        test_csv = SPLITS_DIR / "crema_d" / "test.csv"
        
        # Fallback to old location
        if not train_csv.exists():
            train_csv = SPLITS_DIR / "train.csv"
            val_csv = SPLITS_DIR / "val.csv"
            test_csv = SPLITS_DIR / "test.csv"
    
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Train split not found: {train_csv}\n"
            "Please run data processing scripts first."
        )
    
    # Create datasets
    train_dataset = EmotionDataset(train_csv, label2id)
    val_dataset = EmotionDataset(val_csv, label2id) if val_csv.exists() else None
    test_dataset = EmotionDataset(test_csv, label2id) if test_csv.exists() else None
    
    print(f"Train: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Val: {len(val_dataset)} samples")
    if test_dataset:
        print(f"Test: {len(test_dataset)} samples")
    
    # Create dataloaders - reduce num_workers for MPS to save memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(NUM_WORKERS, 2),  # Reduce workers for MPS
        collate_fn=collate_fn,
        pin_memory=False,  # Disable pin_memory for MPS
    )
    # Use much smaller batch size for validation to avoid MPS memory issues
    val_batch_size = min(args.batch_size, 8)  # Cap at 8 for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing for validation to save memory
        collate_fn=collate_fn,
    ) if val_dataset else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    ) if test_dataset else None
    
    # Create model - use config from checkpoint if available, otherwise use defaults
    checkpoint_data = None
    if args.resume_from and Path(args.resume_from).exists():
        checkpoint_path = Path(args.resume_from)
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "config" in checkpoint_data:
            ckpt_config = checkpoint_data["config"]
            print(f"\n📋 Using config from checkpoint:")
            print(f"   {ckpt_config}")
            # Use checkpoint config directly - it has all the values we need
            cfg = CNNLSTMConfig(
                num_labels=ckpt_config.get("num_labels", len(CANONICAL_EMOTIONS)),
                feature_dim=ckpt_config.get("feature_dim", 37),
                cnn_channels=ckpt_config.get("cnn_channels", 64),
                lstm_hidden=ckpt_config.get("lstm_hidden", 128),
                lstm_layers=ckpt_config.get("lstm_layers", 2),
                dropout=ckpt_config.get("dropout", 0.3),
                use_batch_norm=ckpt_config.get("use_batch_norm", True),
            )
        else:
            # Fallback to default config
            cfg = CNNLSTMConfig(
                num_labels=len(CANONICAL_EMOTIONS),
                feature_dim=config.CNNLSTM_CONFIG["feature_dim"]
            )
    else:
        cfg = CNNLSTMConfig(
            num_labels=len(CANONICAL_EMOTIONS),
            feature_dim=config.CNNLSTM_CONFIG["feature_dim"]
        )
    model = CNNLSTMEmotionModel(cfg)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load from checkpoint if specified (transfer learning)
    start_epoch = 1
    best_val_f1 = 0.0
    if checkpoint_data is not None:
        print(f"\n🔄 Loading checkpoint from: {checkpoint_path}")
        # Load with strict=False to handle any minor architecture differences
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
        if missing_keys:
            print(f"  ⚠ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  ⚠ Unexpected keys (ignored): {len(unexpected_keys)}")
        start_epoch = checkpoint_data.get("epoch", 1) + 1
        best_val_f1 = checkpoint_data.get("best_macro_f1", checkpoint_data.get("val_metrics", {}).get("macro_f1", 0.0))
        print(f"  ✓ Resumed from epoch {checkpoint_data.get('epoch', 'N/A')}")
        print(f"  ✓ Previous best macro-F1: {best_val_f1:.4f}")
        print(f"  ✓ Starting from epoch {start_epoch} (transfer learning)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['macro_f1']:.4f}")
        
        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, criterion, device)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['macro_f1']:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_metrics['macro_f1'])
            
            # Save best model
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                should_save = True
            else:
                should_save = False
        else:
            # No validation set, use training metrics
            val_metrics = train_metrics
            scheduler.step(train_metrics['macro_f1'])
            if train_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = train_metrics['macro_f1']
                should_save = True
            else:
                should_save = False
        
        # Save best model
        if should_save:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.__dict__,
                "id2label": id2label,
                "label2id": label2id,
                "val_metrics": val_metrics,
                "best_macro_f1": best_val_f1,
            }
            checkpoint_path = args.output_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model (F1: {best_val_f1:.4f})")
    
    # Test evaluation
    if test_loader:
        print("\n" + "="*60)
        print("Final Test Evaluation")
        print("="*60)
        
        # Load best model
        checkpoint_path = args.output_dir / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        test_metrics = validate(model, test_loader, criterion, device)
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            test_metrics['labels'],
            test_metrics['predictions'],
            target_names=[id2label[i] for i in range(len(id2label))],
        ))
    
    print(f"\n✓ Training complete! Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

