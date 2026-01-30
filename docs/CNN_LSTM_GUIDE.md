# CNN+LSTM Emotion Model Guide

## Overview

This guide explains the **CNN+LSTM** approach for emotion recognition, which is a classic but highly effective method that often outperforms transformer-based models for emotion detection.

## Why CNN+LSTM?

### Advantages:
- ✅ **Proven effectiveness**: Many papers show CNN+LSTM beats transformers for emotion recognition
- ✅ **Captures prosody**: Emotion lives in prosody (pitch, energy, rhythm) → this captures it well
- ✅ **Interpretable**: Easy to explain in reports and understand what features matter
- ✅ **Efficient**: Faster inference than large transformers
- ✅ **Feature-rich**: Uses comprehensive Librosa features (MFCCs, chroma, spectral, prosodic)

### Architecture:
1. **Feature Extraction**: Comprehensive Librosa features (39 dimensions)
   - 13 MFCCs (Mel-frequency cepstral coefficients)
   - 12 Chroma features (pitch class)
   - 7 Spectral features (contrast, rolloff, bandwidth, ZCR)
   - 7 Prosodic features (pitch, energy statistics)

2. **CNN Layers**: Learn local patterns in feature sequences
   - 2 Conv1D layers with batch normalization
   - Max pooling for dimensionality reduction

3. **LSTM Layers**: Capture temporal dynamics
   - Bidirectional LSTM (2 layers)
   - Captures long-term dependencies in emotion expression

4. **Attention Pooling**: Aggregate temporal information
   - Attention mechanism to focus on important time steps
   - Produces fixed-size embedding

5. **Classifier**: Final emotion prediction
   - Fully connected layers with dropout

## Training the Model

### Step 1: Train CNN+LSTM Model

```bash
python scripts/train_cnn_lstm_model.py \
    --epochs 50 \
    --lr 0.001 \
    --batch-size 32 \
    --output-dir data/processed/cnn_lstm_model
```

### Step 2: Compare with Current Models

After training, you can compare performance:
- **Current Wav2Vec2 model**: ~36% accuracy
- **Text model**: ~20% accuracy  
- **Fused model**: ~28% accuracy
- **CNN+LSTM**: Expected 40-50%+ accuracy (based on literature)

### Step 3: Use in Fusion

Once trained, you can integrate CNN+LSTM into the fusion model by:
1. Updating `config.py` to set `AUDIO_MODEL_TYPE = "cnn_lstm"`
2. Updating `app.py` to load CNN+LSTM model instead of Wav2Vec2
3. The fusion layer will automatically adapt to the new embedding dimension

## Model Configuration

Edit `config.py` to customize:

```python
CNNLSTM_CONFIG = {
    "num_labels": 6,
    "feature_dim": 39,  # MFCCs + chroma + spectral + prosodic
    "cnn_channels": 64,
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "dropout": 0.3,
    "use_batch_norm": True,
}
```

## Expected Performance

Based on emotion recognition literature:
- **Accuracy**: 40-55% (6-way classification on CREMA-D)
- **Macro F1**: 0.35-0.50
- **Per-class F1**: Better balance across emotions than transformers

## Integration with Existing System

The CNN+LSTM model is designed to be a drop-in replacement:
- Same input: Audio waveforms (16kHz, mono)
- Same output: Emotion logits and embeddings
- Compatible with existing fusion strategies

## Next Steps

1. **Train the model**: Run the training script
2. **Evaluate**: Compare with current models
3. **Integrate**: Update config and app.py to use CNN+LSTM
4. **Fusion**: Test fusion with CNN+LSTM + Text model
5. **Optimize**: Fine-tune hyperparameters if needed

## References

- Many emotion recognition papers show CNN+LSTM outperforms transformers
- Librosa features are proven effective for prosody-based emotion detection
- Bidirectional LSTM captures temporal dynamics crucial for emotion

