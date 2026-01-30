"""
Evaluate the trained audio emotion model on the test set
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import pandas as pd

import config
from src.audio_emotion_dataset import CREMAAudioEmotionDataset, collate_fn
from src.audio_emotion_model import AudioEmotionConfig, DistilHuBERTEmotionModel


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    cfg_dict = checkpoint["config"]
    cfg = AudioEmotionConfig(**cfg_dict)
    
    # Create model
    model = DistilHuBERTEmotionModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    id2label = checkpoint["id2label"]
    label2id = checkpoint["label2id"]
    
    print(f"Model loaded. Labels: {list(id2label.values())}")
    return model, id2label, label2id


@torch.no_grad()
def evaluate_test_set(
    model: DistilHuBERTEmotionModel,
    test_loader: DataLoader,
    device: torch.device,
    id2label: dict,
):
    """Evaluate model on test set"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    print("\nEvaluating on test set...")
    for batch_idx, batch in enumerate(test_loader):
        waveforms = batch["waveforms"].to(device)
        prosodics = batch["prosodics"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(waveforms, prosodics)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    # Per-class F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    
    # Classification report
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(
        all_labels, all_preds, target_names=label_names, output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "label_names": label_names,
    }


def print_results(metrics: dict):
    """Print evaluation results"""
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1: {metrics['weighted_f1']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for i, label in enumerate(metrics["label_names"]):
        f1 = metrics["per_class_f1"][i]
        print(f"  {label:12s}: {f1:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        metrics["labels"],
        metrics["predictions"],
        target_names=metrics["label_names"]
    ))
    
    print(f"\nConfusion Matrix:")
    print(f"Rows = True labels, Columns = Predicted labels")
    print(f"{'':12s}", end="")
    for label in metrics["label_names"]:
        print(f"{label:8s}", end="")
    print()
    for i, label in enumerate(metrics["label_names"]):
        print(f"{label:12s}", end="")
        for j in range(len(metrics["label_names"])):
            print(f"{metrics['confusion_matrix'][i, j]:8d}", end="")
        print()
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio emotion model on test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.PROCESSED_DATA_DIR / "audio_emotion_model" / "best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save results to CSV",
    )
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model, id2label, label2id = load_model(checkpoint_path, device)
    
    # Load test dataset
    test_split_path = config.SPLITS_DIR / "test.csv"
    if not test_split_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_split_path}")
    
    print(f"\nLoading test dataset from {test_split_path}")
    test_dataset = CREMAAudioEmotionDataset(
        csv_path=test_split_path,
        label2id=label2id,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"Test set size: {len(test_dataset)} samples")
    
    # Evaluate
    metrics = evaluate_test_set(model, test_loader, device, id2label)
    
    # Print results
    print_results(metrics)
    
    # Save results if requested
    if args.output:
        results_df = pd.DataFrame({
            "true_label": [metrics["label_names"][l] for l in metrics["labels"]],
            "predicted_label": [metrics["label_names"][p] for p in metrics["predictions"]],
        })
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

