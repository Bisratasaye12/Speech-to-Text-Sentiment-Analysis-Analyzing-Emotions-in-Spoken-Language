"""
Training script for MELD dataset
Uses the processed MELD dataset to train the audio emotion recognition model
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess
import sys
import config

# MELD dataset paths
MELD_DATA_DIR = config.DATA_DIR / "meld"
MELD_SPLITS_DIR = MELD_DATA_DIR / "splits"
MELD_OUTPUT_DIR = MELD_DATA_DIR / "models"

# Create output directory
MELD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paths to MELD splits
train_csv = MELD_SPLITS_DIR / "train.csv"
val_csv = MELD_SPLITS_DIR / "val.csv"
test_csv = MELD_SPLITS_DIR / "test.csv"

# Check if splits exist
if not train_csv.exists():
    print(f"❌ Error: Training split not found at {train_csv}")
    print("Please run scripts/process_meld.py first to process the dataset.")
    sys.exit(1)

if not val_csv.exists():
    print(f"❌ Error: Validation split not found at {val_csv}")
    sys.exit(1)

print("=" * 60)
print("Training Audio Emotion Model on MELD Dataset")
print("=" * 60)
print(f"\nTrain CSV: {train_csv}")
print(f"Val CSV: {val_csv}")
print(f"Output directory: {MELD_OUTPUT_DIR}")
print("\nStarting training...\n")

# Build command to run training (use current Python interpreter)
cmd = [
    sys.executable, "scripts/train_audio_emotion_improved.py",
    "--train_csv", str(train_csv),
    "--val_csv", str(val_csv),
    "--batch_size", "16",
    "--epochs", "50",
    "--lr", "2e-4",
    "--use_amp",
    "--output_dir", str(MELD_OUTPUT_DIR),
]

# Run training
try:
    subprocess.run(cmd, check=True)
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nModel saved to: {MELD_OUTPUT_DIR}")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Training failed with error code {e.returncode}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    sys.exit(0)

