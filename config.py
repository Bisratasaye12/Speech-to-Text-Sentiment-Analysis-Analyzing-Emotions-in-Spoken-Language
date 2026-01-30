"""
Configuration file for Speech-to-Text Sentiment Analysis Project
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUDIO_DIR = PROCESSED_DATA_DIR / "audio"
SPLITS_DIR = DATA_DIR / "splits"

# Source audio directory (CREMA-D)
AUDIO_WAV_DIR = PROJECT_ROOT / "AudioWAV"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, AUDIO_DIR, SPLITS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Audio preprocessing settings (Whisper-optimized)
AUDIO_CONFIG = {
    "target_sample_rate": 16000,  # Whisper's native sample rate
    "mono": True,  # Convert to mono
    "normalize": True,  # Peak normalization to [-1, 1]
    "trim_silence": False,
    "min_duration": 0.5,  # Minimum duration in seconds
    "max_duration": 30.0,  # Maximum duration in seconds (Whisper limit)
}

# CREMA-D / core emotion mapping (6-way)
EMOTION_MAPPING = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

# Canonical 6-way emotion set used across the project
CANONICAL_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
]

# Intensity mapping
INTENSITY_MAPPING = {
    "HI": "high",
    "LO": "low",
    "MD": "medium",
    "XX": "unspecified"
}

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Whisper model settings
WHISPER_MODEL = "small"  # Options: tiny, base, small, medium, large

# Processing settings
BATCH_SIZE = 32  # For batch processing
NUM_WORKERS = 4  # For parallel processing (adjust based on CPU cores)

# GoEmotions / text fine-tuning settings
GOEMOTIONS_CONFIG = {
    "dataset_name": "go_emotions",
    "subset": "raw",  # standard GoEmotions subset
}


