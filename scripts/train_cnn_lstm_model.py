"""
Train CNN+LSTM Emotion Model
-----------------------------

This script trains a CNN+LSTM model for emotion recognition using comprehensive
Librosa features (MFCCs, chroma, spectral, prosodic).

This classic approach often outperforms transformer-based models for emotion recognition.
"""

import argparse
import json
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
        
        # Pre-resolve absolute paths
        self.paths: List[Path] = []
        for p in self.df[self.audio_col].tolist():
            path = Path(p)
            if not path.is_absolute():
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
    
    for features, labels in tqdm(dataloader, desc="Training"):
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
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            labels = labels.to(device)
            
            logits, _ = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
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
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create label mappings
    label2id = {label: i for i, label in enumerate(CANONICAL_EMOTIONS)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Load data from split CSVs
    print("Loading data from split CSVs...")
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    ) if val_dataset else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    ) if test_dataset else None
    
    # Create model
    cfg = CNNLSTMConfig(
        num_labels=len(CANONICAL_EMOTIONS),
        feature_dim=config.CNNLSTM_CONFIG["feature_dim"]
    )
    model = CNNLSTMEmotionModel(cfg)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
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

