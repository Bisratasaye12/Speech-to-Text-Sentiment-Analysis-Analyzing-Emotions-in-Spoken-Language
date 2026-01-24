"""
Test script to verify Whisper transcription on a few sample files
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
    Test transcription on a small sample
    """
    print("=" * 60)
    print("Testing Whisper Transcription")
    print("=" * 60)
    
    # Load metadata
    metadata_path = config.PROCESSED_DATA_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)
    
    # Test on first 5 files
    test_df = df.head(5).copy()
    print(f"\nTesting on {len(test_df)} sample files...")
    
    # Initialize transcriber
    print(f"\nInitializing Whisper model: {config.WHISPER_MODEL}")
    transcriber = WhisperTranscriber(model_name=config.WHISPER_MODEL)
    
    # Transcribe
    print("\nTranscribing sample files...")
    result_df = transcriber.transcribe_dataframe(
        test_df,
        audio_path_col='processed_filepath',
        output_col='transcription',
        language='en',
        show_progress=True
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Transcription Results:")
    print("=" * 60)
    
    for idx, row in result_df.iterrows():
        print(f"\n[{row.get('emotion', 'unknown')}] {row.get('filename', 'unknown')}")
        print(f"  Transcription: {row.get('transcription', 'N/A')}")
        print(f"  Language: {row.get('transcription_language', 'N/A')}")
        print(f"  No speech prob: {row.get('transcription_no_speech_prob', 0.0):.3f}")
        print(f"  Success: {row.get('transcription_success', False)}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

