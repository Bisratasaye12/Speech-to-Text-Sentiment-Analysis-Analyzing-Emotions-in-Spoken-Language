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
AUDIO_WAV_DIR = DATA_DIR / "AudioWAV"

# Flask uploads (for API interface - Yord734)
UPLOAD_FOLDER = PROCESSED_DATA_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac"}
MAX_CONTENT_LENGTH_MB = 50  # Max upload size in MB

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

# Audio Model Selection
# Options: "wav2vec2" (transformer-based) or "cnn_lstm" (classic CNN+LSTM)
AUDIO_MODEL_TYPE = "cnn_lstm"  # Using CNN+LSTM model (56.40% accuracy vs 36% with Wav2Vec2)

# CNN+LSTM Model Configuration
CNNLSTM_CONFIG = {
    "num_labels": len(CANONICAL_EMOTIONS),
    "feature_dim": 37,  # 13 MFCCs + 12 chroma + 7 spectral_contrast + 1 rolloff + 1 bandwidth + 1 ZCR + 2 prosodic (f0, rms)
    "cnn_channels": 64,
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "dropout": 0.3,
    "use_batch_norm": True,
}

# Multimodal Fusion Model settings
FUSION_CONFIG = {
    "fusion_type": "late",  # Options: 'early', 'late', 'attention', 'weighted'
    "fusion_hidden_dim": 512,  # Hidden dimension for fusion layers
    "dropout": 0.3,  # Dropout probability for fusion layers
    "text_embedding_dim": 768,  # RoBERTa-base hidden size (fixed)
    # Fusion weights (for late/weighted fusion)
    # Can be: "auto" (performance-based), "equal" (0.5/0.5), or [audio_weight, text_weight]
    "fusion_weights": "auto",  # "auto", "equal", or [float, float] e.g. [0.7, 0.3]
    # Minimum text model accuracy to use in fusion (if below this, use audio-only)
    "min_text_accuracy": 0.25,  # If text accuracy < 25%, fallback to audio-only
    # Note: audio_embedding_dim is determined dynamically from audio model config
    # Note: num_labels is determined from CANONICAL_EMOTIONS
}

# Multi-Dataset Configuration
# Enable/disable datasets and specify their directories
DATASET_CONFIG = {
    "crema_d": {
        "enabled": True,
        "audio_dir": AUDIO_WAV_DIR,
    },
    "tess": {
        "enabled": True,  # TESS dataset enabled
        "audio_dir": RAW_DATA_DIR / "TESS",
    },
    "iemocap": {
        "enabled": True,  # IEMOCAP dataset enabled
        "audio_dir": RAW_DATA_DIR / "IEMOCAP",
    },
}

# Dataset Balancing Configuration
# Controls how samples are balanced across datasets and emotions
BALANCING_CONFIG = {
    # Strategy options:
    # - "equal_per_dataset": Sample equal number from each dataset
    # - "equal_per_emotion": Sample equal number per emotion across all datasets (RECOMMENDED for imbalanced data)
    # - "proportional": Maintain natural dataset ratios but cap maximum
    # - "stratified_per_dataset": Create stratified splits within each dataset, then combine
    "strategy": "equal_per_emotion",  # Changed to balance emotions equally
    
    # Maximum samples per dataset (None = no limit, or int e.g., 5000)
    # Only applies to "proportional" strategy
    "max_samples_per_dataset": None,
    
    # Target samples per emotion class (caps all emotions to this size)
    # Set to 800 to match the smallest emotion class size
    "min_samples_per_emotion": 800,
    
    # Preserve original test split from CREMA-D if available
    # If True, uses existing CREMA-D test split and only balances train/val
    "preserve_test_split": False,
}


