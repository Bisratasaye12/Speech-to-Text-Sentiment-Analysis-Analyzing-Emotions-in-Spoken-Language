# Speech-to-Text Sentiment Analysis: Analyzing Emotions in Spoken Language

NLP course project for analyzing emotions in spoken language using Whisper (STT) and multimodal sentiment analysis.

## Project Overview

This project implements a complete pipeline for:
1. **Speech-to-Text**: Using OpenAI Whisper to transcribe audio
2. **Text Sentiment Analysis**: Using RoBERTa for emotion classification
3. **Audio Sentiment Analysis**: Using audio features for emotion detection
4. **Multimodal Fusion**: Combining text and audio predictions

## Dataset

**CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
- 7,442 audio clips
- 6 emotions: Anger, Disgust, Fear, Happy, Neutral, Sad
- 91 actors with diverse demographics

## Setup

### 1. Create Virtual Environment

```bash
# Using Python 3.10 (recommended)
python3.10 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import whisper; import torch; import librosa; print('All packages installed successfully!')"
```

## Project Structure

```
.
├── .venv/                  # Virtual environment
├── AudioWAV/               # CREMA-D dataset (raw audio files)
├── data/
│   ├── processed/
│   │   ├── audio/          # Preprocessed audio files
│   │   └── metadata.csv    # Full dataset metadata
│   └── splits/
│       ├── train.csv       # Training set (70%)
│       ├── val.csv         # Validation set (15%)
│       └── test.csv        # Test set (15%)
├── src/
│   ├── data_loader.py      # CREMA-D data loading
│   └── preprocessor.py     # Audio preprocessing
├── scripts/
│   └── process_crema_d.py  # Main processing script
├── config.py               # Configuration
├── requirements.txt        # Dependencies
└── README.md
```

## Usage

### Process CREMA-D Dataset

Run the main processing script to:
1. Load metadata from audio filenames
2. Preprocess all audio files (Whisper-optimized)
3. Create train/val/test splits (70/15/15)

```bash
python scripts/process_crema_d.py
```

This will:
- Parse all audio filenames to extract emotion labels
- Preprocess audio files (resample to 16kHz, normalize, etc.)
- Save processed files to `data/processed/audio/`
- Create stratified splits and save to `data/splits/`

### Load Processed Data

```python
import pandas as pd
from pathlib import Path

# Load splits
train_df = pd.read_csv("data/splits/train.csv")
val_df = pd.read_csv("data/splits/val.csv")
test_df = pd.read_csv("data/splits/test.csv")

# Access processed audio files
audio_path = train_df.iloc[0]['processed_filepath']
```

## Configuration

Edit `config.py` to adjust:
- Audio preprocessing settings
- Data split ratios
- Whisper model size
- Processing batch size

## Next Steps

1. ✅ Data & Preprocessing (Current)
2. ⏳ Whisper Transcription Pipeline
3. ⏳ RoBERTa Fine-tuning (LoRA)
4. ⏳ Audio Emotion Model
5. ⏳ Multimodal Fusion
6. ⏳ Flask API Interface

## Requirements

- Python 3.10 or 3.11
- macOS/Windows/Linux
- ~8GB RAM recommended
- ~5GB disk space for processed data

## License

This is a course project. CREMA-D dataset has its own license terms.

