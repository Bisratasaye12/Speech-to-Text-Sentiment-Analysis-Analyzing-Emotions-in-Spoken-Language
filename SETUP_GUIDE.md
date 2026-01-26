# Setup Guide

This guide provides comprehensive instructions for setting up the Speech-to-Text Sentiment Analysis project from scratch, including system requirements, dependencies, and important notes about data portability.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [System-Level Dependencies](#system-level-dependencies)
3. [Python Environment Setup](#python-environment-setup)
4. [Python Package Installation](#python-package-installation)
5. [Dataset Setup](#dataset-setup)
6. [Important Notes About Data Files](#important-notes-about-data-files)
7. [Running the Pipeline](#running-the-pipeline)
8. [Troubleshooting](#troubleshooting)
9. [Moving to Another Environment](#moving-to-another-environment)

---

## System Requirements

### Operating System
- **macOS** (Intel or Apple Silicon)
- **Linux** (Ubuntu 20.04+ recommended)
- **Windows** (Windows 10+ with WSL2 recommended)

### Hardware
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: ~10GB free space (for dataset + processed data + models)
- **GPU**: Optional but recommended for faster processing
  - macOS: MPS (Metal Performance Shaders) support
  - Linux/Windows: CUDA-capable GPU (NVIDIA)

### Python Version
- **Python 3.10** (recommended)
- **Python 3.11** (also supported)
- **Python 3.9** (may work but not tested)

---

## System-Level Dependencies

These tools must be installed **before** setting up the Python environment. They are required for building and running certain Python packages.

### macOS (using Homebrew)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required system tools
brew install ffmpeg          # Audio/video processing (required by Whisper)
brew install cmake           # Build tool (required by llvmlite)
brew install llvm@20         # LLVM compiler (required by llvmlite, version 20 specifically)
```

**Important Notes for macOS:**
- `llvm@20` is required (not the latest version) due to `llvmlite` compatibility
- If you have `llvm` (version 21) installed, unlink it first:
  ```bash
  brew unlink llvm
  brew install llvm@20
  ```

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt-get update

# Install required system tools
sudo apt-get install -y \
    ffmpeg \
    cmake \
    build-essential \
    llvm-14 \
    libllvm-14-dev
```

### Windows

1. **FFmpeg**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   - Extract and add to PATH
   - Or use: `choco install ffmpeg` (if using Chocolatey)

2. **CMake**: Download from [https://cmake.org/download/](https://cmake.org/download/)
   - Or use: `choco install cmake`

3. **LLVM**: Download from [https://llvm.org/builds/](https://llvm.org/builds/)
   - Or use: `choco install llvm`

### Verify System Dependencies

```bash
# Check ffmpeg
ffmpeg -version

# Check cmake
cmake --version

# Check llvm (macOS)
llvm-config --version  # Should show version 20.x

# Check llvm (Linux)
llvm-config-14 --version
```

---

## Python Environment Setup

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd /path/to/Speech-to-Text-Sentiment-Analysis-Analyzing-Emotions-in-Spoken-Language

# Create virtual environment with Python 3.10
python3.10 -m venv .venv

# If Python 3.10 is not available, try:
# python3 -m venv .venv
```

### 2. Activate Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

### 3. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

---

## Python Package Installation

### Install from requirements.txt

```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

This will install:
- **Core ML/Audio**: torch, torchaudio, openai-whisper, transformers, librosa, soundfile
- **Data Processing**: pandas, numpy, scikit-learn, scipy
- **Utilities**: tqdm, python-dotenv
- **Development**: jupyter, ipykernel
- **Future**: peft (for LoRA), flask (for API)

### Verify Installation

```bash
python -c "import whisper; import torch; import librosa; import transformers; print('✓ All packages installed successfully!')"
```

### Expected Installation Time
- First-time installation: 10-30 minutes (depending on internet speed)
- Package downloads: ~2-3GB total

---

## Dataset Setup

### CREMA-D Dataset

1. **Download CREMA-D** from the official source
2. **Extract** the audio files to the project directory
3. **Ensure structure**:
   ```
   Speech-to-Text-Sentiment-Analysis-Analyzing-Emotions-in-Spoken-Language/
   ├── AudioWAV/
   │   ├── 1001_DFA_ANG_XX.wav
   │   ├── 1001_DFA_DIS_XX.wav
   │   └── ... (7,442 files total)
   ```

### Verify Dataset

```bash
# Count audio files
ls AudioWAV/*.wav | wc -l
# Should show ~7,442 files
```

---

## Important Notes About Data Files

### ⚠️ CSV Files Contain Local Filepaths

**CRITICAL**: All CSV files generated by the pipeline (`metadata.csv`, `train.csv`, `val.csv`, `test.csv`) contain **absolute filepaths** specific to your machine.

**Example of filepaths in CSV:**
```csv
filename,processed_filepath,...
1001_DFA_ANG_XX.wav,/Users/username/Desktop/.../data/processed/audio/1001_DFA_ANG_XX.wav,...
```

### What This Means

1. **Not Portable**: CSV files from one machine will **NOT work** on another machine
2. **Path Mismatches**: If you move the project or run on a different machine, filepaths will be incorrect
3. **Solution**: You must **regenerate** the CSV files on each new environment (see [Moving to Another Environment](#moving-to-another-environment))

### Files Affected

- `data/processed/metadata.csv` - Contains all metadata + filepaths
- `data/splits/train.csv` - Training split with filepaths
- `data/splits/val.csv` - Validation split with filepaths
- `data/splits/test.csv` - Test split with filepaths

### Why Absolute Paths?

- Ensures scripts can find files regardless of working directory
- Prevents path resolution issues
- Makes debugging easier

---

## Running the Pipeline

### Step 1: Data Preprocessing

```bash
# Activate virtual environment first
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate    # Windows

# Run preprocessing
python scripts/process_crema_d.py
```

**What it does:**
- Parses CREMA-D filenames to extract metadata (emotion, actor, sentence, intensity)
- Preprocesses all audio files (resample to 16kHz, normalize, convert to mono)
- Creates train/val/test splits (70/15/15 ratio)
- Saves processed audio to `data/processed/audio/`
- Generates CSV files with **local filepaths**

**Expected time:** 30-60 minutes (depending on CPU)

### Step 2: Transcription (Train & Val Only)

```bash
python scripts/transcribe_audio.py
```

**What it does:**
- Loads train and validation splits
- Transcribes audio using Whisper model
- Updates `train.csv` and `val.csv` with transcriptions
- **Leaves test split untouched** (for final evaluation)

**Expected time:** 2-6 hours (depending on model size and device)
- Whisper "base" model: ~2-3 hours
- Whisper "small" model: ~4-6 hours

### Step 3: Exploratory Data Analysis

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_EDA_CREMA_Dataset.ipynb
```

The notebook includes:
- Dataset overview and statistics
- Emotion distribution analysis
- Transcription quality checks
- Data splits validation

**Note:** The first cell will automatically install missing packages if needed.

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'matplotlib'`

**Solution:**
```bash
# Install missing packages
pip install matplotlib seaborn

# Or run the first cell in the EDA notebook
```

### Issue: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`

**Solution:**
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### Issue: `CMake Error` or `llvmlite` build fails

**Solution:**
```bash
# macOS - Install correct LLVM version
brew unlink llvm  # If version 21 is installed
brew install llvm@20

# Linux - Install LLVM 14
sudo apt-get install llvm-14 libllvm-14-dev
```

### Issue: `NotImplementedError: Could not run 'aten::empty.memory_format' with arguments from the 'SparseMPS' backend`

**Solution:**
- This is a known MPS (Metal) backend issue on macOS
- The code is already configured to use CPU instead of MPS
- If you see this error, check `src/transcriber.py` - device should be set to `"cpu"`

### Issue: `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`

**Solution:**
```bash
# Update torch and torchaudio
pip install --upgrade torch>=2.2.0 torchaudio>=2.2.0
```

### Issue: CSV files show incorrect filepaths

**Solution:**
- Regenerate CSV files by running `scripts/process_crema_d.py` again
- Ensure you're running from the project root directory
- Check that `config.py` has correct `PROJECT_ROOT` path

### Issue: Transcription shows "no speech" for all files

**Solution:**
- Verify `ffmpeg` is installed and in PATH: `ffmpeg -version`
- Check that audio files are valid: `ffprobe <audio_file>`
- Ensure audio preprocessing completed successfully

---

## Moving to Another Environment

If you need to run this project on a different machine or share it with collaborators:

### Option 1: Regenerate All Data (Recommended)

1. **Copy the project** (excluding generated data):
   ```bash
   # Copy project structure
   cp -r Speech-to-Text-Sentiment-Analysis-Analyzing-Emotions-in-Spoken-Language /new/location/
   
   # Copy dataset
   cp -r AudioWAV /new/location/
   ```

2. **On the new machine:**
   - Follow [System-Level Dependencies](#system-level-dependencies)
   - Follow [Python Environment Setup](#python-environment-setup)
   - Follow [Python Package Installation](#python-package-installation)

3. **Regenerate all data:**
   ```bash
   # This will create new CSV files with correct filepaths
   python scripts/process_crema_d.py
   python scripts/transcribe_audio.py
   ```

### Option 2: Update Filepaths in CSV (Not Recommended)

If you must reuse CSV files:

1. **Find and replace** filepaths in all CSV files:
   ```bash
   # Example: Replace old path with new path
   sed -i 's|/old/path|/new/path|g' data/processed/metadata.csv
   sed -i 's|/old/path|/new/path|g' data/splits/*.csv
   ```

2. **Verify** filepaths are correct:
   ```python
   import pandas as pd
   df = pd.read_csv('data/splits/train.csv')
   import os
   print(os.path.exists(df.iloc[0]['processed_filepath']))
   ```

**Warning:** This method is error-prone and not recommended. Regenerating is safer.

### What to Share vs. What Not to Share

**Share (via Git):**
- ✅ Source code (`src/`, `scripts/`, `config.py`)
- ✅ `requirements.txt`
- ✅ `README.md`, `SETUP_GUIDE.md`
- ✅ Jupyter notebooks (without outputs)
- ✅ `.gitignore`

**Do NOT Share (add to `.gitignore`):**
- ❌ `.venv/` (virtual environment)
- ❌ `data/` (processed data and CSV files)
- ❌ `AudioWAV/` (dataset - too large)
- ❌ Model checkpoints
- ❌ `__pycache__/`, `*.pyc`

---

## Project Structure Reference

```
Speech-to-Text-Sentiment-Analysis-Analyzing-Emotions-in-Spoken-Language/
├── .venv/                      # Virtual environment (not in Git)
├── AudioWAV/                   # CREMA-D dataset (not in Git)
│   └── *.wav                   # 7,442 audio files
├── data/                       # Generated data (not in Git)
│   ├── processed/
│   │   ├── audio/              # Preprocessed audio files
│   │   └── metadata.csv        # Full metadata (local filepaths!)
│   └── splits/
│       ├── train.csv           # Training split (local filepaths!)
│       ├── val.csv             # Validation split (local filepaths!)
│       └── test.csv            # Test split (local filepaths!)
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_loader.py          # CREMA-D data loading
│   ├── preprocessor.py         # Audio preprocessing
│   └── transcriber.py          # Whisper transcription
├── scripts/                     # Executable scripts
│   ├── process_crema_d.py      # Main preprocessing pipeline
│   ├── transcribe_audio.py     # Transcription (train/val only)
│   └── test_transcription.py   # Test transcription
├── notebooks/                   # Jupyter notebooks
│   └── 01_EDA_CREMA_Dataset.ipynb
├── config.py                   # Configuration
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project overview
└── SETUP_GUIDE.md             # This file
```

---

## Quick Start Checklist

- [ ] Install system dependencies (ffmpeg, cmake, llvm)
- [ ] Create Python virtual environment (`.venv`)
- [ ] Activate virtual environment
- [ ] Install Python packages (`pip install -r requirements.txt`)
- [ ] Verify installation
- [ ] Place CREMA-D dataset in `AudioWAV/` folder
- [ ] Run `python scripts/process_crema_d.py`
- [ ] Run `python scripts/transcribe_audio.py`
- [ ] Open `notebooks/01_EDA_CREMA_Dataset.ipynb` for analysis

---

## Additional Resources

- **Whisper Documentation**: [https://github.com/openai/whisper](https://github.com/openai/whisper)
- **CREMA-D Dataset**: [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- **PyTorch Installation**: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## Support

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting](#troubleshooting) section
2. Verify all system dependencies are installed
3. Ensure virtual environment is activated
4. Check that filepaths in CSV files are correct for your system
5. Regenerate data files if moving to a new environment

---

**Last Updated**: Based on project state as of current implementation
**Python Version**: 3.10 recommended
**Tested On**: macOS (Intel), macOS (Apple Silicon)

