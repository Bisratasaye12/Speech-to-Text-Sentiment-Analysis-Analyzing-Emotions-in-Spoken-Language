"""
MELD Dataset Processing Script
- Downloads MELD dataset from Kaggle
- Extracts audio from video files
- Filters to 6 emotions (anger, disgust, fear, happy, neutral, sad)
- Balances the dataset
- Preprocesses audio files
- Creates train/val/test splits
"""
import sys
import os
from pathlib import Path
import subprocess
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import librosa
import soundfile as sf
from collections import Counter

import config
from src.preprocessor import AudioPreprocessor


# MELD emotion mapping to our 6 emotions
MELD_EMOTION_MAPPING = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",  # Map joy to happy
    "neutral": "neutral",
    "sadness": "sad",  # Map sadness to sad
    "surprise": None  # Drop surprise
}

# Target emotions (6 emotions we want)
TARGET_EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]

# MELD dataset paths
MELD_DATA_DIR = config.DATA_DIR / "meld"
# Check for actual downloaded location
if (MELD_DATA_DIR / "MELD-RAW" / "MELD.Raw").exists():
    MELD_RAW_DIR = MELD_DATA_DIR / "MELD-RAW" / "MELD.Raw"
else:
    MELD_RAW_DIR = MELD_DATA_DIR / "raw"
MELD_AUDIO_DIR = MELD_DATA_DIR / "audio_extracted"
MELD_PROCESSED_DIR = MELD_DATA_DIR / "processed"
MELD_SPLITS_DIR = MELD_DATA_DIR / "splits"

# Create directories (don't create MELD_RAW_DIR as it should already exist)
for dir_path in [MELD_AUDIO_DIR, MELD_PROCESSED_DIR, MELD_SPLITS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def check_kaggle_installed() -> bool:
    """Check if kaggle is installed"""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_dataset_exists():
    """Check if MELD dataset already exists"""
    print("=" * 60)
    print("Checking for MELD Dataset")
    print("=" * 60)
    
    # Check for dataset in various possible locations
    possible_locations = [
        MELD_DATA_DIR / "MELD-RAW" / "MELD.Raw",
        MELD_DATA_DIR / "raw",
        MELD_RAW_DIR,
    ]
    
    for location in possible_locations:
        train_csv = location / "train" / "train_sent_emo.csv"
        if train_csv.exists():
            print(f"\n✓ Dataset found at {location}")
            return True
    
    print(f"\n⚠ Dataset not found. Please ensure MELD dataset is downloaded.")
    print(f"  Expected location: {MELD_RAW_DIR}")
    print(f"  Or: {MELD_DATA_DIR / 'MELD-RAW' / 'MELD.Raw'}")
    return False


def extract_audio_from_video(video_path: Path, output_audio_path: Path) -> bool:
    """
    Extract audio from video file using ffmpeg
    
    Args:
        video_path: Path to video file
        output_audio_path: Path to save extracted audio
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if ffmpeg is available
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: ffmpeg is not installed.")
        print("Please install it using: brew install ffmpeg (on macOS)")
        return False
    
    try:
        # Extract audio using ffmpeg
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # Sample rate 16kHz
            "-ac", "1",  # Mono
            "-y",  # Overwrite output file
            str(output_audio_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Check if it's just a warning (ffmpeg sometimes returns non-zero for warnings)
            if "error" in result.stderr.lower() and "does not contain any stream" not in result.stderr.lower():
                return False
            # If output file doesn't exist, it failed
            if not output_audio_path.exists():
                return False
        
        return output_audio_path.exists()
        
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {str(e)}")
        return False


def load_meld_metadata() -> pd.DataFrame:
    """
    Load MELD metadata from CSV files
    
    Returns:
        Combined DataFrame with all metadata
    """
    print("\n[1/6] Loading MELD metadata...")
    
    # MELD has train, dev, and test splits in specific locations
    split_configs = [
        ("train", MELD_RAW_DIR / "train" / "train_sent_emo.csv"),
        ("dev", MELD_RAW_DIR / "dev_sent_emo.csv"),  # dev CSV is in root, not in dev/ folder
        ("test", MELD_RAW_DIR / "test_sent_emo.csv"),  # test CSV is in root, not in test/ folder
    ]
    
    all_data = []
    
    for split_name, csv_path in split_configs:
        if not csv_path.exists():
            print(f"  ⚠ Warning: {csv_path} not found, skipping {split_name} split")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            df['split'] = split_name
            all_data.append(df)
            print(f"  ✓ Loaded {len(df)} samples from {split_name} split")
        except Exception as e:
            print(f"  ⚠ Error loading {csv_path}: {str(e)}")
    
    if not all_data:
        raise FileNotFoundError("No MELD metadata files found!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  ✓ Total samples: {len(combined_df)}")
    
    return combined_df


def map_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map MELD emotions to our 6 target emotions
    
    Args:
        df: DataFrame with MELD data
        
    Returns:
        DataFrame with mapped emotions
    """
    print("\n[2/6] Mapping emotions...")
    
    # MELD typically has 'Emotion' column (case-insensitive check)
    emotion_col = None
    for col in df.columns:
        if col.lower() in ['emotion', 'sentiment', 'label']:
            emotion_col = col
            break
    
    if emotion_col is None:
        raise ValueError("Could not find emotion column in MELD data")
    
    print(f"  Using column: {emotion_col}")
    
    # Show original emotion distribution
    original_emotions = df[emotion_col].value_counts()
    print("\n  Original emotion distribution:")
    for emotion, count in original_emotions.items():
        print(f"    - {emotion}: {count}")
    
    # Map emotions
    df['emotion_mapped'] = df[emotion_col].str.lower().map(MELD_EMOTION_MAPPING)
    
    # Filter out None (dropped emotions like surprise)
    before_count = len(df)
    df = df[df['emotion_mapped'].notna()].copy()
    after_count = len(df)
    
    print(f"\n  ✓ Filtered from {before_count} to {after_count} samples")
    print(f"  ✓ Dropped {before_count - after_count} samples with non-target emotions")
    
    # Rename column
    df['emotion'] = df['emotion_mapped']
    df = df.drop(columns=['emotion_mapped'])
    
    # Show new emotion distribution
    new_emotions = df['emotion'].value_counts()
    print("\n  Mapped emotion distribution:")
    for emotion in TARGET_EMOTIONS:
        count = new_emotions.get(emotion, 0)
        print(f"    - {emotion}: {count}")
    
    return df


def find_video_files(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find video files corresponding to utterances
    
    Args:
        df: DataFrame with MELD metadata
        
    Returns:
        DataFrame with video file paths added
    """
    print("\n[3/6] Finding video files...")
    
    # MELD video files are in split-specific directories
    video_dirs = {
        "train": MELD_RAW_DIR / "train" / "train_splits",
        "dev": MELD_RAW_DIR / "dev" / "dev_splits_complete",
        "test": MELD_RAW_DIR / "test" / "output_repeated_splits_test",
    }
    
    # Build video file mapping: video filename -> full path
    video_files = {}
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    for split_name, video_dir in video_dirs.items():
        if video_dir.exists():
            for video_file in video_dir.rglob("*"):
                if video_file.suffix.lower() in video_extensions:
                    stem = video_file.stem
                    video_files[stem] = video_file
            print(f"  ✓ Found videos in {split_name}: {video_dir}")
    
    if not video_files:
        raise FileNotFoundError("Could not find any video files in MELD dataset")
    
    print(f"  ✓ Total video files found: {len(video_files)}")
    
    # MELD video files are named: dia{Dialogue_ID}_utt{Utterance_ID}.mp4
    def find_video_path(row):
        # Get Dialogue_ID and Utterance_ID
        if 'Dialogue_ID' not in row.index or 'Utterance_ID' not in row.index:
            return None
        
        dia_id = int(row['Dialogue_ID']) if pd.notna(row['Dialogue_ID']) else None
        utt_id = int(row['Utterance_ID']) if pd.notna(row['Utterance_ID']) else None
        
        if dia_id is None or utt_id is None:
            return None
        
        # Try exact match: dia{dia_id}_utt{utt_id}
        video_name = f"dia{dia_id}_utt{utt_id}"
        if video_name in video_files:
            return video_files[video_name]
        
        # Try with different formats
        patterns = [
            f"dia{dia_id}_utt{utt_id}",
            f"dia{str(dia_id).zfill(3)}_utt{utt_id}",
            f"dia{dia_id}_utt{str(utt_id).zfill(2)}",
        ]
        
        for pattern in patterns:
            if pattern in video_files:
                return video_files[pattern]
        
        return None
    
    df['video_path'] = df.apply(find_video_path, axis=1)
    
    found_count = df['video_path'].notna().sum()
    print(f"  ✓ Matched {found_count} / {len(df)} utterances to video files")
    
    if found_count == 0:
        raise FileNotFoundError("Could not match any utterances to video files!")
    
    # Filter to only rows with video files
    df = df[df['video_path'].notna()].copy()
    
    return df


def extract_audio_files(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract audio from all video files
    
    Args:
        df: DataFrame with video paths
        
    Returns:
        DataFrame with audio paths added
    """
    print("\n[4/5] Extracting audio from videos...")
    
    audio_paths = []
    failed_extractions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Extracting audio"):
        video_path = Path(row['video_path'])
        
        # Create output audio path
        audio_filename = f"{video_path.stem}.wav"
        audio_path = MELD_AUDIO_DIR / audio_filename
        
        # Skip if already extracted
        if audio_path.exists():
            audio_paths.append(str(audio_path))
            continue
        
        # Extract audio
        if extract_audio_from_video(video_path, audio_path):
            audio_paths.append(str(audio_path))
        else:
            audio_paths.append(None)
            failed_extractions.append(video_path)
    
    df['audio_path'] = audio_paths
    
    # Filter out failed extractions
    before_count = len(df)
    df = df[df['audio_path'].notna()].copy()
    after_count = len(df)
    
    if failed_extractions:
        print(f"  ⚠ Warning: Failed to extract audio from {len(failed_extractions)} videos")
    
    print(f"  ✓ Extracted audio from {after_count} / {before_count} videos")
    
    return df


def balance_dataset(df: pd.DataFrame, min_samples_per_class: int = None) -> pd.DataFrame:
    """
    Balance the dataset by undersampling majority classes
    
    Args:
        df: DataFrame with emotion labels
        min_samples_per_class: Minimum samples per class (default: minimum class count)
        
    Returns:
        Balanced DataFrame
    """
    print("\n[5/6] Balancing dataset...")
    
    emotion_counts = df['emotion'].value_counts()
    print("\n  Emotion distribution before balancing:")
    for emotion in TARGET_EMOTIONS:
        count = emotion_counts.get(emotion, 0)
        print(f"    - {emotion}: {count}")
    
    if min_samples_per_class is None:
        # Use minimum class count
        min_samples_per_class = emotion_counts.min()
    
    print(f"\n  Balancing to {min_samples_per_class} samples per class...")
    
    balanced_dfs = []
    for emotion in TARGET_EMOTIONS:
        emotion_df = df[df['emotion'] == emotion].copy()
        
        if len(emotion_df) > min_samples_per_class:
            # Undersample
            emotion_df = emotion_df.sample(
                n=min_samples_per_class,
                random_state=config.RANDOM_SEED
            )
        elif len(emotion_df) < min_samples_per_class:
            print(f"  ⚠ Warning: {emotion} has only {len(emotion_df)} samples (less than {min_samples_per_class})")
        
        balanced_dfs.append(emotion_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    emotion_counts_after = balanced_df['emotion'].value_counts()
    print("\n  Emotion distribution after balancing:")
    for emotion in TARGET_EMOTIONS:
        count = emotion_counts_after.get(emotion, 0)
        print(f"    - {emotion}: {count}")
    
    print(f"\n  ✓ Balanced dataset: {len(balanced_df)} total samples")
    
    return balanced_df


def preprocess_audio_files(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess all audio files
    
    Args:
        df: DataFrame with audio paths
        
    Returns:
        DataFrame with processed audio paths
    """
    print("\n[6/6] Preprocessing audio files...")
    
    preprocessor = AudioPreprocessor()
    processed_paths = []
    valid_flags = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Preprocessing"):
        audio_path = Path(row['audio_path'])
        
        # Create output path
        output_filename = f"meld_{audio_path.stem}.wav"
        output_path = MELD_PROCESSED_DIR / output_filename
        
        # Skip if already processed
        if output_path.exists():
            processed_paths.append(str(output_path))
            valid_flags.append(True)
            continue
        
        try:
            # Preprocess audio
            audio, sr, metadata = preprocessor.preprocess(
                str(audio_path),
                str(output_path),
                save=True
            )
            
            processed_paths.append(str(output_path))
            valid_flags.append(metadata['is_valid'])
            
        except Exception as e:
            print(f"\n  ⚠ Error processing {audio_path}: {str(e)}")
            processed_paths.append(None)
            valid_flags.append(False)
    
    df['processed_audio_path'] = processed_paths
    df['is_valid'] = valid_flags
    
    # Filter out invalid files
    before_count = len(df)
    df = df[df['is_valid']].copy()
    after_count = len(df)
    
    print(f"  ✓ Processed {after_count} / {before_count} valid audio files")
    
    return df


def create_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test splits
    
    Args:
        df: DataFrame with all data
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("\n[7/7] Creating train/val/test splits...")
    
    # Use stratified split to maintain emotion distribution
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_RATIO,
        stratify=df['emotion'],
        random_state=config.RANDOM_SEED
    )
    
    val_size_adjusted = config.VAL_RATIO / (1 - config.TEST_RATIO)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df['emotion'],
        random_state=config.RANDOM_SEED
    )
    
    print(f"  ✓ Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  ✓ Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  ✓ Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Show emotion distribution in each split
    print("\n  Split emotion distribution:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n    {split_name}:")
        emotion_counts = split_df['emotion'].value_counts().sort_index()
        for emotion, count in emotion_counts.items():
            pct = count / len(split_df) * 100
            print(f"      - {emotion}: {count} ({pct:.1f}%)")
    
    return train_df, val_df, test_df


def save_metadata_and_splits(df: pd.DataFrame, train_df: pd.DataFrame, 
                            val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save metadata and splits to CSV files"""
    print("\n[8/8] Saving metadata and splits...")
    
    # Prepare final metadata DataFrame
    metadata_df = df[['emotion', 'processed_audio_path']].copy()
    metadata_df.columns = ['emotion', 'filepath']
    metadata_df['filename'] = metadata_df['filepath'].apply(lambda x: Path(x).name)
    
    # Add duration
    durations = []
    for filepath in metadata_df['filepath']:
        try:
            duration = librosa.get_duration(path=filepath)
            durations.append(duration)
        except:
            durations.append(None)
    metadata_df['duration'] = durations
    
    # Save full metadata
    metadata_path = MELD_PROCESSED_DIR / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"  ✓ Saved metadata to {metadata_path}")
    
    # Prepare split DataFrames
    train_meta = train_df[['emotion', 'processed_audio_path']].copy()
    train_meta.columns = ['emotion', 'filepath']
    train_meta['filename'] = train_meta['filepath'].apply(lambda x: Path(x).name)
    
    val_meta = val_df[['emotion', 'processed_audio_path']].copy()
    val_meta.columns = ['emotion', 'filepath']
    val_meta['filename'] = val_meta['filepath'].apply(lambda x: Path(x).name)
    
    test_meta = test_df[['emotion', 'processed_audio_path']].copy()
    test_meta.columns = ['emotion', 'filepath']
    test_meta['filename'] = test_meta['filepath'].apply(lambda x: Path(x).name)
    
    # Save splits
    train_path = MELD_SPLITS_DIR / "train.csv"
    val_path = MELD_SPLITS_DIR / "val.csv"
    test_path = MELD_SPLITS_DIR / "test.csv"
    
    train_meta.to_csv(train_path, index=False)
    val_meta.to_csv(val_path, index=False)
    test_meta.to_csv(test_path, index=False)
    
    print(f"  ✓ Saved train split to {train_path}")
    print(f"  ✓ Saved validation split to {val_path}")
    print(f"  ✓ Saved test split to {test_path}")


def main():
    """Main processing pipeline"""
    print("=" * 60)
    print("MELD Dataset Processing Pipeline")
    print("=" * 60)
    
    # Step 1: Check if dataset exists
    if not check_dataset_exists():
        print("\n❌ Dataset not found. Please download the MELD dataset first.")
        print("  You can download it from: https://www.kaggle.com/datasets/zaber666/meld-dataset/data")
        return
    
    # Step 2: Load metadata
    try:
        df = load_meld_metadata()
    except Exception as e:
        print(f"\n❌ Error loading metadata: {str(e)}")
        return
    
    # Step 3: Map emotions
    try:
        df = map_emotions(df)
    except Exception as e:
        print(f"\n❌ Error mapping emotions: {str(e)}")
        return
    
    # Step 4: Find video files
    try:
        df = find_video_files(df)
    except Exception as e:
        print(f"\n❌ Error finding video files: {str(e)}")
        return
    
    # Step 5: Extract audio
    try:
        df = extract_audio_files(df)
    except Exception as e:
        print(f"\n❌ Error extracting audio: {str(e)}")
        return
    
    # Step 6: Balance dataset
    try:
        df = balance_dataset(df)
    except Exception as e:
        print(f"\n❌ Error balancing dataset: {str(e)}")
        return
    
    # Step 7: Preprocess audio
    try:
        df = preprocess_audio_files(df)
    except Exception as e:
        print(f"\n❌ Error preprocessing audio: {str(e)}")
        return
    
    # Step 8: Create splits
    try:
        train_df, val_df, test_df = create_splits(df)
    except Exception as e:
        print(f"\n❌ Error creating splits: {str(e)}")
        return
    
    # Step 9: Save metadata and splits
    try:
        save_metadata_and_splits(df, train_df, val_df, test_df)
    except Exception as e:
        print(f"\n❌ Error saving metadata: {str(e)}")
        return
    
    print("\n" + "=" * 60)
    print("MELD Dataset Processing Complete!")
    print("=" * 60)
    print(f"\nProcessed files saved to: {MELD_PROCESSED_DIR}")
    print(f"Metadata saved to: {MELD_PROCESSED_DIR / 'metadata.csv'}")
    print(f"Splits saved to: {MELD_SPLITS_DIR}")


if __name__ == "__main__":
    main()

