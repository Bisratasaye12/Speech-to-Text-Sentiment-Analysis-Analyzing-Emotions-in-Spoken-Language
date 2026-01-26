"""
Script to transcribe all CREMA-D audio files using Whisper
"""
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.transcriber import WhisperTranscriber
import config


def main():
    """
    Main transcription pipeline
    Transcribes only train and validation splits. Test split is left untouched.
    """
    print("=" * 60)
    print("Whisper Transcription Pipeline")
    print("=" * 60)
    print("NOTE: Only transcribing TRAIN and VAL splits.")
    print("      TEST split will be transcribed at the end.")
    print("=" * 60)
    
    # Load split files
    print("\n[1/5] Loading split files...")
    train_path = config.SPLITS_DIR / "train.csv"
    val_path = config.SPLITS_DIR / "val.csv"
    test_path = config.SPLITS_DIR / "test.csv"
    
    if not train_path.exists() or not val_path.exists():
        print(f"Error: Split files not found in {config.SPLITS_DIR}")
        print("Please run scripts/process_crema_d.py first")
        return
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"  ✓ Loaded train split: {len(train_df)} files")
    print(f"  ✓ Loaded val split: {len(val_df)} files")
    
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        print(f"  ✓ Test split found: {len(test_df)} files (will NOT be transcribed)")
    else:
        print(f"  ⚠ Test split not found (will be created later)")
    
    # Combine train and val for transcription
    df_to_transcribe = pd.concat([train_df, val_df], ignore_index=True)
    print(f"\n  Total files to transcribe: {len(df_to_transcribe)} (train + val)")
    
    # Check for existing transcriptions
    existing_transcriptions = 0
    if 'transcription' in df_to_transcribe.columns:
        existing_transcriptions = df_to_transcribe['transcription'].notna().sum()
        if existing_transcriptions > 0:
            print(f"\n  ℹ Found {existing_transcriptions} existing transcriptions")
            print(f"  ℹ Will resume from checkpoint (skipping already transcribed files)")
    
    # Setup checkpoint file
    checkpoint_dir = config.PROCESSED_DATA_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / "transcription_checkpoint.csv"
    checkpoint_path_relative = checkpoint_file.relative_to(config.PROJECT_ROOT)
    
    # Load checkpoint if it exists
    if checkpoint_file.exists():
        print(f"\n  ℹ Found checkpoint file: {checkpoint_path_relative}")
        print(f"  ℹ Loading checkpoint to resume transcription...")
        checkpoint_df = pd.read_csv(checkpoint_file)
        
        # Check how many transcriptions are in the checkpoint
        if 'transcription' in checkpoint_df.columns:
            checkpoint_transcribed = (
                (checkpoint_df['transcription'].notna()) & 
                (checkpoint_df['transcription'] != '')
            ).sum()
            print(f"  ℹ Checkpoint contains {checkpoint_transcribed} transcriptions out of {len(checkpoint_df)} rows")
        
        # Merge checkpoint data with current dataframe
        if 'transcription' in checkpoint_df.columns:
            merge_cols = ['filename'] + [col for col in checkpoint_df.columns if col.startswith('transcription')]
            df_to_transcribe = df_to_transcribe.merge(
                checkpoint_df[merge_cols],
                on='filename',
                how='left',
                suffixes=('', '_checkpoint')
            )
            # Use checkpoint values where available (checkpoint takes priority)
            for col in merge_cols:
                if col != 'filename' and f'{col}_checkpoint' in df_to_transcribe.columns:
                    # Prioritize checkpoint values: if checkpoint has a value, use it; otherwise keep original
                    df_to_transcribe[col] = df_to_transcribe[f'{col}_checkpoint'].fillna(df_to_transcribe[col])
                    df_to_transcribe = df_to_transcribe.drop(columns=[f'{col}_checkpoint'])
            df_to_transcribe = df_to_transcribe.loc[:, ~df_to_transcribe.columns.duplicated()]
            
            # Report how many transcriptions were loaded
            loaded_transcriptions = (
                (df_to_transcribe['transcription'].notna()) & 
                (df_to_transcribe['transcription'] != '')
            ).sum()
            print(f"  ✓ Checkpoint loaded: {loaded_transcriptions} transcriptions available in current dataframe")
    
    # Initialize transcriber
    print(f"\n[2/5] Initializing Whisper transcriber...")
    print(f"  Model: {config.WHISPER_MODEL}")
    transcriber = WhisperTranscriber(model_name=config.WHISPER_MODEL)
    
    # Count files that need transcription
    # Initialize transcription columns if they don't exist
    if 'transcription' not in df_to_transcribe.columns:
        df_to_transcribe['transcription'] = ''
    if 'transcription_success' not in df_to_transcribe.columns:
        df_to_transcribe['transcription_success'] = False
    
    # A file is considered transcribed if it has non-empty text AND success=True
    is_transcribed = (
        (df_to_transcribe['transcription'].notna()) & 
        (df_to_transcribe['transcription'] != '') &
        (df_to_transcribe['transcription_success'] == True)
    )
    needs_transcription = (~is_transcribed).sum()
    already_transcribed = is_transcribed.sum()
    
    print(f"\n[3/5] Transcribing {needs_transcription} audio files (train + val)...")
    print(f"  (Skipping {already_transcribed} already transcribed files)")
    print(f"  Checkpoint: {checkpoint_path_relative} (saved every 50 files)")
    print("  This may take a while depending on model size and device...")
    
    df_transcribed = transcriber.transcribe_dataframe(
        df_to_transcribe,
        audio_path_col='processed_filepath',
        output_col='transcription',
        language='en',
        show_progress=True,
        skip_existing=True,  # Skip already transcribed files
        save_checkpoint=str(checkpoint_file),  # Save checkpoint periodically
        checkpoint_interval=50  # Save every 50 files
    )
    
    # Save final checkpoint
    df_transcribed.to_csv(checkpoint_file, index=False)
    print(f"\n  ✓ Final checkpoint saved: {checkpoint_path_relative}")
    
    # Statistics
    print(f"\n[4/5] Transcription Statistics:")
    successful = df_transcribed['transcription_success'].sum() if 'transcription_success' in df_transcribed.columns else len(df_transcribed)
    print(f"  ✓ Successfully transcribed: {successful}/{len(df_transcribed)} files")
    
    if 'transcription' in df_transcribed.columns:
        # Calculate average transcription length
        valid_transcriptions = df_transcribed[df_transcribed['transcription'].notna() & (df_transcribed['transcription'] != '')]
        if len(valid_transcriptions) > 0:
            avg_length = valid_transcriptions['transcription'].str.len().mean()
            print(f"  ✓ Average transcription length: {avg_length:.1f} characters")
            
            # Show sample transcriptions
            print(f"\n  Sample transcriptions:")
            for idx, row in valid_transcriptions.head(3).iterrows():
                emotion = row.get('emotion', 'unknown')
                text = row['transcription'][:80] + "..." if len(row['transcription']) > 80 else row['transcription']
                print(f"    [{emotion}] {text}")
    
    # Split back into train and val
    print(f"\n[5/5] Updating split files with transcriptions...")
    
    # Get filenames for each split
    train_filenames = set(train_df['filename'])
    val_filenames = set(val_df['filename'])
    
    # Split transcribed dataframe back
    train_transcribed = df_transcribed[df_transcribed['filename'].isin(train_filenames)].copy()
    val_transcribed = df_transcribed[df_transcribed['filename'].isin(val_filenames)].copy()
    
    # Update train split
    train_path = config.SPLITS_DIR / "train.csv"
    train_transcribed.to_csv(train_path, index=False)
    print(f"  ✓ Updated train.csv ({len(train_transcribed)} files)")
    
    # Update val split
    val_path = config.SPLITS_DIR / "val.csv"
    val_transcribed.to_csv(val_path, index=False)
    print(f"  ✓ Updated val.csv ({len(val_transcribed)} files)")
    
    # Test split remains untouched
    if test_path.exists():
        print(f"  ✓ Test split left untouched ({len(test_df)} files)")
    
    # Also update main metadata file if it exists
    metadata_path = config.PROCESSED_DATA_DIR / "metadata.csv"
    if metadata_path.exists():
        print(f"\n  Updating main metadata file...")
        metadata_df = pd.read_csv(metadata_path)
        
        # Merge transcriptions from transcribed data
        if 'transcription' in df_transcribed.columns:
            merge_cols = ['filename'] + [col for col in df_transcribed.columns if col.startswith('transcription')]
            metadata_df = metadata_df.merge(
                df_transcribed[merge_cols],
                on='filename',
                how='left',
                suffixes=('', '_new')
            )
            # Remove duplicate columns if any
            metadata_df = metadata_df.loc[:, ~metadata_df.columns.duplicated()]
            metadata_df.to_csv(metadata_path, index=False)
            print(f"    ✓ Updated metadata.csv")
    
    print("\n" + "=" * 60)
    print("Transcription complete!")
    print("=" * 60)
    splits_path = config.SPLITS_DIR.relative_to(config.PROJECT_ROOT)
    print(f"\nTrain and Val splits updated: {splits_path}/")
    print("Test split left untouched (will be transcribed at the end)")


if __name__ == "__main__":
    main()

