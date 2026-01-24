# Notebooks Directory

This directory contains Jupyter notebooks for exploratory data analysis and experimentation.

## Notebooks

### 01_EDA_CREMA_Dataset.ipynb
Exploratory Data Analysis of the CREMA-D dataset:
- Dataset overview and statistics
- Emotion distribution analysis
- Audio characteristics (duration, features)
- Actor distribution
- Transcription analysis (if available)
- Data splits validation
- Sample audio analysis

## Usage

1. Start Jupyter Lab:
```bash
source .venv/bin/activate
jupyter lab
```

2. Open the notebook from the Jupyter interface

3. Make sure to run cells in order (some depend on previous cells)

## Requirements

All required packages are in `requirements.txt`. The notebooks use:
- pandas, numpy for data manipulation
- matplotlib, seaborn for visualization
- librosa for audio analysis
- Project modules (src/) for data loading

