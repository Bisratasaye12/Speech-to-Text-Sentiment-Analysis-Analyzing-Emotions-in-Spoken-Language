# Multi-Dataset Integration Guide

## Overview

This project now supports training on multiple emotion recognition datasets:
- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset (~7,000 files)
- **TESS**: Toronto Emotional Speech Set (~2,800 files)
- **IEMOCAP**: Interactive Emotional Dyadic Motion Capture (~10,000 files)

Combining these datasets provides:
- **14+ speakers** (vs 2 in CREMA-D alone)
- **200+ sentences** (vs ~12 in CREMA-D alone)
- **Natural conversations** (from IEMOCAP)
- **Better generalization** and reduced overfitting

## Quick Start

### 1. Download Datasets

**TESS:**
- Download from: [Kaggle - Toronto Emotional Speech Set](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- Extract to: `data/raw/TESS/`

**IEMOCAP:**
- Download from: [USC IEMOCAP](https://sail.usc.edu/iemocap/) (requires registration)
- Extract to: `data/raw/IEMOCAP/`

### 2. Enable Datasets

Edit `config.py`:

```python
DATASET_CONFIG = {
    "crema_d": {
        "enabled": True,
        "audio_dir": AUDIO_WAV_DIR,
    },
    "tess": {
        "enabled": True,  # Enable TESS
        "audio_dir": RAW_DATA_DIR / "TESS",
    },
    "iemocap": {
        "enabled": True,  # Enable IEMOCAP
        "audio_dir": RAW_DATA_DIR / "IEMOCAP",
    },
}
```

### 3. Prepare Combined Dataset

```bash
python scripts/prepare_multi_dataset.py
```

This will:
- Load metadata from all enabled datasets
- Apply balancing strategy
- Preprocess audio files
- Create stratified train/val/test splits
- Save to `data/splits/multi_dataset/`

### 4. Train Model

```bash
# CNN+LSTM model
python scripts/train_cnn_lstm_model.py

# Or improved model with augmentation
python scripts/train_audio_emotion_improved.py --use-augmentation --use-class-weights
```

The training scripts automatically detect and use multi-dataset splits if available.

## Dataset Balancing Strategies

Configure in `config.py`:

```python
BALANCING_CONFIG = {
    "strategy": "stratified_per_dataset",  # Recommended
    "max_samples_per_dataset": None,
    "min_samples_per_emotion": 100,
    "preserve_test_split": False,
}
```

### Available Strategies

1. **`stratified_per_dataset`** (Recommended)
   - Creates stratified splits within each dataset
   - Combines splits maintaining emotion distribution
   - Best for generalization

2. **`equal_per_dataset`**
   - Samples equal number from each dataset
   - Preserves dataset diversity
   - May underutilize larger datasets

3. **`equal_per_emotion`**
   - Samples equal number per emotion across all datasets
   - Ensures balanced emotion distribution
   - May lose dataset-specific patterns

4. **`proportional`**
   - Maintains natural dataset ratios
   - Caps maximum per dataset (if `max_samples_per_dataset` set)
   - Balances diversity and size

## File Structure

```
data/
├── raw/
│   ├── TESS/                    # TESS dataset
│   └── IEMOCAP/                 # IEMOCAP dataset
│
├── splits/
│   ├── crema_d/                 # CREMA-D only splits
│   └── multi_dataset/            # Combined dataset splits
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── statistics.json
│
└── processed/
    └── audio/
        ├── CREMA-D_*.wav        # Processed CREMA-D
        ├── TESS_*.wav            # Processed TESS
        └── IEMOCAP_*.wav         # Processed IEMOCAP
```

## Implementation Details

### New Modules

1. **`src/multi_dataset_loader.py`**
   - `TESSDataLoader`: Parses TESS filenames and extracts metadata
   - `IEMOCAPDataLoader`: Loads IEMOCAP annotations and metadata
   - `MultiDatasetLoader`: Orchestrates loading from all datasets

2. **`src/dataset_balancer.py`**
   - `DatasetBalancer`: Implements 4 balancing strategies
   - Creates stratified splits with dataset awareness

3. **`src/audio_emotion_dataset.py`**
   - `MultiDatasetAudioEmotionDataset`: Unified dataset class
   - Handles different file path structures transparently

4. **`scripts/prepare_multi_dataset.py`**
   - Main processing pipeline
   - Combines all steps: loading, balancing, preprocessing, splitting

### Updated Modules

- `config.py`: Added `DATASET_CONFIG` and `BALANCING_CONFIG`
- `scripts/train_cnn_lstm_model.py`: Auto-detects multi-dataset splits
- `scripts/train_audio_emotion_improved.py`: Auto-detects multi-dataset splits

## Emotion Mapping

All datasets are mapped to canonical 6-way emotions:

- **anger**: anger, frustration (IEMOCAP)
- **disgust**: disgust
- **fear**: fear
- **happy**: happy, excitement (IEMOCAP), joy
- **neutral**: neutral, surprise
- **sad**: sad

## Expected Results

### Dataset Sizes (after balancing)

- **Strategy 4 (Stratified per dataset)**:
  - Train: ~12,000-15,000 files
  - Val: ~2,000-3,000 files
  - Test: ~2,000-3,000 files
  - Total: ~16,000-21,000 files

### Accuracy Improvements

- **Current (CREMA-D only)**: 56.40%
- **With TESS**: 60-65%
- **With TESS + IEMOCAP**: 65-75%
- **With all improvements + multi-dataset**: 75-85%

## Command-Line Options

### `prepare_multi_dataset.py`

```bash
python scripts/prepare_multi_dataset.py \
    --crema-dir PATH \
    --tess-dir PATH \
    --iemocap-dir PATH \
    --balance-strategy stratified_per_dataset \
    --max-samples 5000 \
    --skip-preprocessing \
    --output-dir data/splits/multi_dataset
```

### Training Scripts

Both training scripts automatically detect multi-dataset splits. You can also specify:

```bash
python scripts/train_cnn_lstm_model.py \
    --csv-path data/splits/multi_dataset/train.csv
```

## Troubleshooting

### Dataset Not Found

If a dataset is not found:
1. Check that the dataset is downloaded and in the correct directory
2. Verify `DATASET_CONFIG` in `config.py` has correct paths
3. Enable the dataset: `DATASET_CONFIG["tess"]["enabled"] = True`

### Path Resolution Issues

If audio files are not found:
- Check that preprocessing completed successfully
- Verify file paths in CSV files
- Ensure `processed_filepath` column exists in splits

### Balancing Issues

If balancing produces too few samples:
- Reduce `min_samples_per_emotion` in `BALANCING_CONFIG`
- Try a different balancing strategy
- Check that all datasets have sufficient samples per emotion

## Statistics

After preparation, check `data/splits/multi_dataset/statistics.json` for:
- Total samples per dataset
- Emotion distribution
- Split composition
- Balancing metrics

## Backward Compatibility

- Existing CREMA-D-only code continues to work
- Training scripts fallback to CREMA-D splits if multi-dataset not found
- All existing scripts and notebooks remain functional

## Next Steps

1. Download TESS and IEMOCAP datasets
2. Enable datasets in `config.py`
3. Run `prepare_multi_dataset.py`
4. Train model with combined dataset
5. Compare results with CREMA-D-only baseline

## References

- **CREMA-D**: [GitHub](https://github.com/CheyneyComputerScience/CREMA-D)
- **TESS**: [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- **IEMOCAP**: [USC](https://sail.usc.edu/iemocap/)

