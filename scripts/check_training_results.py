#!/usr/bin/env python3
"""Check training results from checkpoint files."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import config

def main():
    checkpoint_dir = config.PROCESSED_DATA_DIR / "models" / "checkpoints"
    best_model_path = checkpoint_dir / "best_model.pt"
    
    if not best_model_path.exists():
        print(f"❌ Best model checkpoint not found at: {best_model_path}")
        return
    
    print(f"📊 Loading checkpoint from: {best_model_path.relative_to(config.PROJECT_ROOT)}\n")
    
    checkpoint = torch.load(best_model_path, map_location='cpu')
    
    print("=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best Macro-F1: {checkpoint.get('best_macro_f1', 0.0):.4f}")
    print(f"Epochs without improvement: {checkpoint.get('epochs_without_improvement', 0)}")
    print()
    
    # Check all checkpoint files for epoch-by-epoch results
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    
    if checkpoint_files:
        print("=" * 60)
        print("EPOCH-BY-EPOCH RESULTS")
        print("=" * 60)
        
        for ckpt_file in checkpoint_files:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            epoch = ckpt.get('epoch', 'N/A')
            best_f1 = ckpt.get('best_macro_f1', 0.0)
            print(f"Epoch {epoch}: Best Macro-F1 = {best_f1:.4f}")
    
    print()
    print("=" * 60)
    print("NOTE: For detailed accuracy and macro-F1 metrics,")
    print("check the training script output above.")
    print("=" * 60)

if __name__ == "__main__":
    main()

