"""
Quick benchmark to estimate training time per batch/epoch.
"""

import sys
import time
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Suppress expected warnings
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")

# Suppress expected warnings
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.text_dataset import create_datasets_from_csv
from src.text_model import load_roberta_text_model


def benchmark_training_time(
    csv_path: Path,
    batch_size: int = 32,
    num_batches_to_test: int = 10,
    max_length: int = 128,
):
    """Run a quick benchmark to estimate training time."""
    # Auto-detect best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Testing with {num_batches_to_test} batches...\n")

    # Load model
    print("Loading model...", end=" ", flush=True)
    artifacts = load_roberta_text_model(
        model_name="roberta-base",
        use_lora=True,
        freeze_base=True,
    )
    model = artifacts.model.to(device)
    tokenizer = artifacts.tokenizer
    print("✓")

    # Load datasets
    print("Loading datasets...", end=" ", flush=True)
    train_ds, val_ds, label_enc = create_datasets_from_csv(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_length=max_length,
        val_ratio=0.1,
    )
    print("✓")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # Setup optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5, weight_decay=0.01)

    total_steps = len(train_loader) * 3  # Assume 3 epochs for scheduler
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Benchmark training
    print("\n" + "=" * 60)
    print("BENCHMARKING TRAINING TIME")
    print("=" * 60)

    model.train()
    train_times = []
    for i, batch in enumerate(train_loader):
        if i >= num_batches_to_test:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        start_time = time.time()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        elapsed = time.time() - start_time
        train_times.append(elapsed)
        print(f"  Batch {i+1}/{num_batches_to_test}: {elapsed:.3f}s", end="\r")

    print()  # New line after progress

    # Benchmark validation
    print("\nBENCHMARKING VALIDATION TIME")
    print("=" * 60)

    model.eval()
    val_times = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches_to_test:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            start_time = time.time()

            outputs = model(**batch)

            elapsed = time.time() - start_time
            val_times.append(elapsed)
            print(f"  Batch {i+1}/{num_batches_to_test}: {elapsed:.3f}s", end="\r")

    print()  # New line after progress

    # Calculate statistics
    avg_train_time = sum(train_times) / len(train_times)
    avg_val_time = sum(val_times) / len(val_times)

    train_batches_per_epoch = len(train_loader)
    val_batches_per_epoch = len(val_loader)

    train_time_per_epoch = avg_train_time * train_batches_per_epoch
    val_time_per_epoch = avg_val_time * val_batches_per_epoch
    total_time_per_epoch = train_time_per_epoch + val_time_per_epoch

    # Print results
    print("\n" + "=" * 60)
    print("ESTIMATED TRAINING TIME")
    print("=" * 60)
    print(f"\nDataset size:")
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples: {len(val_ds):,}")
    print(f"  Train batches/epoch: {train_batches_per_epoch:,}")
    print(f"  Val batches/epoch: {val_batches_per_epoch:,}")

    print(f"\nTime per batch (averaged over {num_batches_to_test} batches):")
    print(f"  Training: {avg_train_time:.3f}s")
    print(f"  Validation: {avg_val_time:.3f}s")

    print(f"\nTime per epoch:")
    print(f"  Training: {train_time_per_epoch/60:.2f} minutes ({train_time_per_epoch:.1f}s)")
    print(f"  Validation: {val_time_per_epoch/60:.2f} minutes ({val_time_per_epoch:.1f}s)")
    print(f"  Total: {total_time_per_epoch/60:.2f} minutes ({total_time_per_epoch:.1f}s)")

    print(f"\nEstimated time for full training (3 epochs):")
    total_3_epochs = total_time_per_epoch * 3
    print(f"  {total_3_epochs/3600:.2f} hours ({total_3_epochs/60:.1f} minutes)")

    print(f"\nEstimated time for full training (5 epochs):")
    total_5_epochs = total_time_per_epoch * 5
    print(f"  {total_5_epochs/3600:.2f} hours ({total_5_epochs/60:.1f} minutes)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    csv_path = config.RAW_DATA_DIR / "goemotions_6class_train_full.csv"
    benchmark_training_time(
        csv_path=csv_path,
        batch_size=32,
        num_batches_to_test=20,  # Test 20 batches for better average
        max_length=128,
    )

