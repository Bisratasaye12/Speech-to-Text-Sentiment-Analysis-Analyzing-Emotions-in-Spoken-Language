"""
Script to update processed data paths after moving AudioWAV folder
Updates metadata.csv and split files with correct paths
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.data_loader import CREMADataLoader
import config


def update_metadata_paths():
    """Update metadata.csv with correct paths"""
    print("=" * 60)
    print("Updating Processed Data Paths")
    print("=" * 60)
    
    # Step 1: Load fresh metadata from AudioWAV directory
    print("\n[1/3] Loading metadata from AudioWAV directory...")
    loader = CREMADataLoader(audio_dir=config.AUDIO_WAV_DIR)
    
    try:
        df = loader.load_metadata()
        print(f"  ✓ Loaded {len(df)} audio files")
        print(f"  ✓ Found {df['emotion'].nunique()} emotions: {', '.join(df['emotion'].unique())}")
    except Exception as e:
        print(f"  ✗ Error loading metadata: {e}")
        return
    
    # Step 2: Update processed file paths
    print("\n[2/3] Updating processed file paths...")
    
    # Check which processed files exist
    processed_files = []
    missing_files = []
    
    for filename in df['filename']:
        processed_path = config.AUDIO_DIR / filename
        if processed_path.exists():
            processed_files.append(str(processed_path))
        else:
            missing_files.append(filename)
            # Use original filepath if processed doesn't exist
            original_path = df[df['filename'] == filename]['filepath'].iloc[0]
            processed_files.append(original_path)
    
    df['processed_filepath'] = processed_files
    
    if missing_files:
        print(f"  ⚠ Warning: {len(missing_files)} processed files not found, using original files")
        print(f"  ⚠ First 5 missing: {missing_files[:5]}")
    
    # Step 3: Save updated metadata
    print("\n[3/3] Saving updated metadata...")
    metadata_path = config.PROCESSED_DATA_DIR / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"  ✓ Saved updated metadata to {metadata_path}")
    
    # Step 4: Update split files if they exist
    print("\n[4/4] Updating split files...")
    
    for split_name in ['train', 'val', 'test']:
        split_path = config.SPLITS_DIR / f"{split_name}.csv"
        
        if split_path.exists():
            print(f"\n  Updating {split_name}.csv...")
            split_df = pd.read_csv(split_path)
            
            # Get columns to update from df (excluding filename which is the key)
            cols_to_update = [col for col in df.columns if col != 'filename']
            
            # Drop columns that will be updated (except filename)
            cols_to_drop = [col for col in cols_to_update if col in split_df.columns]
            if cols_to_drop:
                split_df = split_df.drop(columns=cols_to_drop)
            
            # Merge with updated metadata
            split_df = split_df.merge(
                df[['filename'] + cols_to_update],
                on='filename',
                how='left'
            )
            
            split_df.to_csv(split_path, index=False)
            print(f"    ✓ Updated {split_name}.csv ({len(split_df)} files)")
        else:
            print(f"  ⚠ {split_name}.csv not found, skipping")
    
    # Show statistics
    print("\n" + "=" * 60)
    print("Update Complete!")
    print("=" * 60)
    
    stats = loader.get_statistics(df)
    print(f"\nDataset Statistics:")
    print(f"  - Total files: {stats['total_files']}")
    print(f"  - Unique actors: {stats['unique_actors']}")
    print(f"  - Unique sentences: {stats['unique_sentences']}")
    print(f"  - Emotions: {stats['emotions']}")
    
    if 'avg_duration' in stats:
        print(f"  - Avg duration: {stats['avg_duration']:.2f}s")
    
    print("\nEmotion Distribution:")
    emotion_dist = loader.get_emotion_distribution(df)
    for emotion, row in emotion_dist.iterrows():
        print(f"  - {emotion}: {row['count']} files ({row['percentage']:.1f}%)")
    
    print(f"\n✓ Metadata saved to: {metadata_path}")
    print(f"✓ Splits saved to: {config.SPLITS_DIR}")


if __name__ == "__main__":
    update_metadata_paths()

