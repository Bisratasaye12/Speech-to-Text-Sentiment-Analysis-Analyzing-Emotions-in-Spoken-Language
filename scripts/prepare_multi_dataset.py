"""
Prepare Combined Multi-Dataset for Audio Emotion Recognition
-----------------------------------------------------------

This script:
1. Loads metadata from CREMA-D, TESS, and IEMOCAP datasets
2. Applies dataset balancing strategy
3. Preprocesses all audio files
4. Creates stratified train/val/test splits
5. Saves splits and statistics
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.multi_dataset_loader import MultiDatasetLoader
from src.dataset_balancer import DatasetBalancer
from src.preprocessor import AudioPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Prepare combined multi-dataset (CREMA-D + TESS + IEMOCAP) for training"
    )
    parser.add_argument(
        "--crema-dir",
        type=str,
        default=None,
        help="Path to CREMA-D audio directory (overrides config)",
    )
    parser.add_argument(
        "--tess-dir",
        type=str,
        default=None,
        help="Path to TESS audio directory (overrides config)",
    )
    parser.add_argument(
        "--iemocap-dir",
        type=str,
        default=None,
        help="Path to IEMOCAP root directory (overrides config)",
    )
    parser.add_argument(
        "--balance-strategy",
        type=str,
        default=None,
        choices=["equal_per_dataset", "equal_per_emotion", "proportional", "stratified_per_dataset"],
        help="Balancing strategy (overrides config)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for proportional strategy)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip audio preprocessing (use existing processed files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for splits (default: data/splits/multi_dataset)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Dataset Preparation Pipeline")
    print("=" * 60)
    
    # Step 1: Determine dataset directories
    crema_dir = Path(args.crema_dir) if args.crema_dir else (
        config.DATASET_CONFIG["crema_d"]["audio_dir"] 
        if config.DATASET_CONFIG["crema_d"]["enabled"] 
        else None
    )
    tess_dir = Path(args.tess_dir) if args.tess_dir else (
        config.DATASET_CONFIG["tess"]["audio_dir"] 
        if config.DATASET_CONFIG["tess"]["enabled"] 
        else None
    )
    iemocap_dir = Path(args.iemocap_dir) if args.iemocap_dir else (
        config.DATASET_CONFIG["iemocap"]["audio_dir"] 
        if config.DATASET_CONFIG["iemocap"]["enabled"] 
        else None
    )
    
    # Step 2: Load metadata from all datasets
    print("\n[1/5] Loading metadata from all datasets...")
    loader = MultiDatasetLoader(
        crema_dir=crema_dir,
        tess_dir=tess_dir,
        iemocap_dir=iemocap_dir,
    )
    
    try:
        combined_df = loader.load_all_datasets()
    except Exception as e:
        print(f"\n❌ Error loading datasets: {e}")
        print("\nPlease ensure:")
        print("  1. Datasets are downloaded and placed in correct directories")
        print("  2. Dataset paths are configured in config.py or provided via arguments")
        print("  3. Enable datasets in config.py: DATASET_CONFIG['tess']['enabled'] = True")
        return 1
    
    # Step 3: Apply dataset balancing
    print("\n[2/5] Applying dataset balancing...")
    balance_strategy = args.balance_strategy or config.BALANCING_CONFIG["strategy"]
    max_samples = args.max_samples or config.BALANCING_CONFIG.get("max_samples_per_dataset")
    min_samples = config.BALANCING_CONFIG.get("min_samples_per_emotion", 100)
    
    balancer = DatasetBalancer(
        strategy=balance_strategy,
        max_samples_per_dataset=max_samples,
        min_samples_per_emotion=min_samples,
        random_state=config.RANDOM_SEED,
    )
    
    balanced_df = balancer.balance(combined_df)
    
    print(f"  Strategy: {balance_strategy}")
    print(f"  Before balancing: {len(combined_df)} files")
    print(f"  After balancing: {len(balanced_df)} files")
    
    # Show statistics
    stats = balancer.get_statistics(balanced_df)
    print(f"\n  Dataset distribution:")
    for dataset, count in stats["datasets"].items():
        print(f"    {dataset}: {count} files")
    print(f"\n  Emotion distribution:")
    for emotion, count in stats["emotions"].items():
        print(f"    {emotion}: {count} files")
    
    # Step 4: Preprocess audio files (if not skipped)
    if not args.skip_preprocessing:
        print("\n[3/5] Preprocessing audio files...")
        preprocessor = AudioPreprocessor()
        
        processed_paths = []
        errors = []
        
        for idx, row in balanced_df.iterrows():
            original_path = Path(row["filepath"])
            dataset = row.get("dataset", "UNKNOWN")
            filename = original_path.name
            
            # Create processed filename with dataset prefix
            processed_filename = f"{dataset}_{filename}"
            processed_path = config.AUDIO_DIR / processed_filename
            
            # Skip if already processed
            if processed_path.exists():
                processed_paths.append(processed_path)
                continue
            
            # Preprocess
            try:
                audio, sr, metadata = preprocessor.preprocess(
                    str(original_path),
                    str(processed_path),
                    save=True,
                )
                if metadata["is_valid"]:
                    processed_paths.append(processed_path)
                else:
                    errors.append(f"{original_path}: Invalid audio")
                    processed_paths.append(None)
            except Exception as e:
                errors.append(f"{original_path}: {str(e)}")
                processed_paths.append(None)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(balanced_df)} files...")
        
        if errors:
            print(f"\n  ⚠ {len(errors)} files had errors during preprocessing")
            for error in errors[:10]:
                print(f"    - {error}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more")
        
        balanced_df["processed_filepath"] = [
            str(p) if p else None for p in processed_paths
        ]
        
        # Filter out invalid files
        before_count = len(balanced_df)
        balanced_df = balanced_df[balanced_df["processed_filepath"].notna()]
        after_count = len(balanced_df)
        
        if before_count != after_count:
            print(f"  ⚠ Filtered {before_count - after_count} invalid files")
    else:
        print("\n[3/5] Skipping preprocessing (using existing processed files)")
        # Assume processed_filepath exists or use filepath
        if "processed_filepath" not in balanced_df.columns:
            balanced_df["processed_filepath"] = balanced_df["filepath"]
    
    # Step 5: Create stratified splits
    print("\n[4/5] Creating stratified train/val/test splits...")
    train_df, val_df, test_df = balancer.create_stratified_splits(
        balanced_df,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
    )
    
    print(f"  Train: {len(train_df)} files")
    print(f"  Val: {len(val_df)} files")
    print(f"  Test: {len(test_df)} files")
    
    # Step 6: Save splits and statistics
    print("\n[5/5] Saving splits and statistics...")
    output_dir = Path(args.output_dir) if args.output_dir else config.SPLITS_DIR / "multi_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV files
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    # Save statistics
    final_stats = {
        "total_samples": len(balanced_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "balancing_strategy": balance_strategy,
        "dataset_distribution": stats["datasets"],
        "emotion_distribution": stats["emotions"],
        "train_emotion_distribution": train_df["emotion"].value_counts().to_dict(),
        "val_emotion_distribution": val_df["emotion"].value_counts().to_dict(),
        "test_emotion_distribution": test_df["emotion"].value_counts().to_dict(),
        "train_dataset_distribution": train_df["dataset"].value_counts().to_dict(),
        "val_dataset_distribution": val_df["dataset"].value_counts().to_dict(),
        "test_dataset_distribution": test_df["dataset"].value_counts().to_dict(),
    }
    
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(final_stats, f, indent=2)
    
    print(f"\n✓ Saved splits to {output_dir}")
    print(f"  - train.csv ({len(train_df)} files)")
    print(f"  - val.csv ({len(val_df)} files)")
    print(f"  - test.csv ({len(test_df)} files)")
    print(f"  - statistics.json")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Preparation Complete!")
    print("=" * 60)
    print(f"\nTotal samples: {len(balanced_df)}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"\nDataset composition:")
    for dataset, count in stats["datasets"].items():
        pct = (count / len(balanced_df)) * 100
        print(f"  {dataset}: {count} ({pct:.1f}%)")
    print(f"\nEmotion composition:")
    for emotion, count in stats["emotions"].items():
        pct = (count / len(balanced_df)) * 100
        print(f"  {emotion}: {count} ({pct:.1f}%)")
    print(f"\n✓ Ready for training!")
    print(f"  Use: python scripts/train_cnn_lstm_model.py --csv-path {output_dir / 'train.csv'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

