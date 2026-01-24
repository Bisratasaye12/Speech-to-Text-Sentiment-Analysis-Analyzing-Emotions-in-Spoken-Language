"""
Script to transcribe all CREMA-D audio files using Whisper
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.transcriber import WhisperTranscriber
import config


def main():
    """
    Main transcription pipeline
    """
    print("=" * 60)
    print("Whisper Transcription Pipeline")
    print("=" * 60)
    
    # Load metadata
    print("\n[1/4] Loading metadata...")
    metadata_path = config.PROCESSED_DATA_DIR / "metadata.csv"
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        print("Please run scripts/process_crema_d.py first")
        return
    
    df = pd.read_csv(metadata_path)
    print(f"  ✓ Loaded {len(df)} audio files")
    
    # Check if transcriptions already exist
    if 'transcription' in df.columns and df['transcription'].notna().any():
        print("\n  ⚠ Warning: Transcriptions already exist in metadata")
        response = input("  Do you want to overwrite? (y/n): ")
        if response.lower() != 'y':
            print("  Cancelled.")
            return
        # Remove existing transcription columns
        cols_to_remove = [col for col in df.columns if col.startswith('transcription')]
        df = df.drop(columns=cols_to_remove)
    
    # Initialize transcriber
    print(f"\n[2/4] Initializing Whisper transcriber...")
    print(f"  Model: {config.WHISPER_MODEL}")
    transcriber = WhisperTranscriber(model_name=config.WHISPER_MODEL)
    
    # Transcribe all files
    print(f"\n[3/4] Transcribing {len(df)} audio files...")
    print("  This may take a while depending on model size and device...")
    
    df = transcriber.transcribe_dataframe(
        df,
        audio_path_col='processed_filepath',
        output_col='transcription',
        language='en',
        show_progress=True
    )
    
    # Statistics
    print(f"\n[4/4] Transcription Statistics:")
    successful = df['transcription_success'].sum() if 'transcription_success' in df.columns else len(df)
    print(f"  ✓ Successfully transcribed: {successful}/{len(df)} files")
    
    if 'transcription' in df.columns:
        # Calculate average transcription length
        valid_transcriptions = df[df['transcription'].notna() & (df['transcription'] != '')]
        if len(valid_transcriptions) > 0:
            avg_length = valid_transcriptions['transcription'].str.len().mean()
            print(f"  ✓ Average transcription length: {avg_length:.1f} characters")
            
            # Show sample transcriptions
            print(f"\n  Sample transcriptions:")
            for idx, row in valid_transcriptions.head(3).iterrows():
                emotion = row.get('emotion', 'unknown')
                text = row['transcription'][:80] + "..." if len(row['transcription']) > 80 else row['transcription']
                print(f"    [{emotion}] {text}")
    
    # Save updated metadata
    print(f"\n[5/5] Saving transcriptions to metadata...")
    output_path = config.PROCESSED_DATA_DIR / "metadata.csv"
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved to {output_path}")
    
    # Also update split files
    print(f"\n  Updating split files with transcriptions...")
    for split_name in ['train', 'val', 'test']:
        split_path = config.SPLITS_DIR / f"{split_name}.csv"
        if split_path.exists():
            split_df = pd.read_csv(split_path)
            # Merge transcriptions from main metadata
            if 'transcription' in df.columns:
                merge_cols = ['filename'] + [col for col in df.columns if col.startswith('transcription')]
                split_df = split_df.merge(
                    df[merge_cols],
                    on='filename',
                    how='left',
                    suffixes=('', '_new')
                )
                # Remove duplicate columns if any
                split_df = split_df.loc[:, ~split_df.columns.duplicated()]
                split_df.to_csv(split_path, index=False)
                print(f"    ✓ Updated {split_name}.csv")
    
    print("\n" + "=" * 60)
    print("Transcription complete!")
    print("=" * 60)
    print(f"\nTranscriptions saved to: {output_path}")
    print(f"Split files updated in: {config.SPLITS_DIR}")


if __name__ == "__main__":
    main()

