"""
Main script to process CREMA-D dataset
- Loads metadata from audio filenames
- Preprocesses all audio files
- Creates train/val/test splits (70/15/15)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.data_loader import CREMADataLoader
from src.preprocessor import AudioPreprocessor
import config


def create_splits(df: pd.DataFrame, test_size: float = 0.15, 
                   val_size: float = 0.15, random_state: int = 42) -> Tuple:
    """
    Create stratified train/val/test splits
    
    Args:
        df: Metadata DataFrame
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['emotion'],
        random_state=random_state
    )
    
    # Second split: separate train and val from remaining data
    # Adjust val_size to account for test split
    val_size_adjusted = val_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df['emotion'],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def main():
    """Main processing pipeline"""
    print("=" * 60)
    print("CREMA-D Dataset Processing Pipeline")
    print("=" * 60)
    
    # Step 1: Load metadata
    print("\n[1/4] Loading metadata from audio filenames...")
    loader = CREMADataLoader()
    df = loader.load_metadata()
    
    print(f"  ✓ Loaded {len(df)} audio files")
    print(f"  ✓ Found {df['emotion'].nunique()} emotions: {', '.join(df['emotion'].unique())}")
    print(f"  ✓ Found {df['actor_id'].nunique()} unique actors")
    
    # Show statistics
    stats = loader.get_statistics(df)
    print("\n  Dataset Statistics:")
    print(f"    - Total files: {stats['total_files']}")
    print(f"    - Unique actors: {stats['unique_actors']}")
    print(f"    - Unique sentences: {stats['unique_sentences']}")
    
    if 'avg_duration' in stats:
        print(f"    - Avg duration: {stats['avg_duration']:.2f}s")
        print(f"    - Duration range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s")
    
    # Show emotion distribution
    emotion_dist = loader.get_emotion_distribution(df)
    print("\n  Emotion Distribution:")
    for emotion, row in emotion_dist.iterrows():
        print(f"    - {emotion}: {row['count']} files ({row['percentage']:.1f}%)")
    
    # Step 2: Preprocess audio files
    print("\n[2/4] Preprocessing audio files...")
    preprocessor = AudioPreprocessor()
    
    # Process all files
    filepaths = [Path(fp) for fp in df['filepath']]
    metadata_list = preprocessor.batch_preprocess(
        filepaths,
        config.AUDIO_DIR,
        verbose=True
    )
    
    # Update DataFrame with preprocessing metadata
    preprocess_df = pd.DataFrame(metadata_list)
    df = df.merge(
        preprocess_df[['original_path', 'duration', 'is_valid']],
        left_on='filepath',
        right_on='original_path',
        how='left'
    )
    
    # Filter out invalid files
    invalid_count = (~df['is_valid']).sum()
    if invalid_count > 0:
        print(f"\n  ⚠ Warning: {invalid_count} files have invalid duration and will be excluded")
        df = df[df['is_valid']].copy()
    
    # Update filepaths to point to processed files
    df['processed_filepath'] = df['filename'].apply(
        lambda x: str(config.AUDIO_DIR / x)
    )
    
    print(f"  ✓ Processed {len(df)} valid audio files")
    
    # Step 3: Create splits
    print("\n[3/4] Creating train/val/test splits (70/15/15)...")
    train_df, val_df, test_df = create_splits(
        df,
        test_size=config.TEST_RATIO,
        val_size=config.VAL_RATIO,
        random_state=config.RANDOM_SEED
    )
    
    print(f"  ✓ Train set: {len(train_df)} files ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  ✓ Validation set: {len(val_df)} files ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  ✓ Test set: {len(test_df)} files ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify stratification
    print("\n  Split Emotion Distribution:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n    {split_name}:")
        emotion_counts = split_df['emotion'].value_counts().sort_index()
        for emotion, count in emotion_counts.items():
            pct = count / len(split_df) * 100
            print(f"      - {emotion}: {count} ({pct:.1f}%)")
    
    # Step 4: Save metadata and splits
    print("\n[4/4] Saving metadata and splits...")
    
    # Save full metadata
    metadata_path = config.PROCESSED_DATA_DIR / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"  ✓ Saved full metadata to {metadata_path}")
    
    # Save splits
    train_path = config.SPLITS_DIR / "train.csv"
    val_path = config.SPLITS_DIR / "val.csv"
    test_path = config.SPLITS_DIR / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  ✓ Saved train split to {train_path}")
    print(f"  ✓ Saved validation split to {val_path}")
    print(f"  ✓ Saved test split to {test_path}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"\nProcessed files saved to: {config.AUDIO_DIR}")
    print(f"Metadata saved to: {config.PROCESSED_DATA_DIR}")
    print(f"Splits saved to: {config.SPLITS_DIR}")


if __name__ == "__main__":
    main()

