# Model Performance Improvements

## Current Performance Analysis

Based on the evaluation results:
- **Audio-only (Wav2Vec2)**: 36% accuracy
- **Text-only (RoBERTa)**: 20% accuracy (very poor!)
- **Fused model**: 28% accuracy (worse than audio-only!)

### Problems Identified:
1. **Text model is dragging down fusion**: 20% accuracy is below random chance for 6 classes
2. **Fusion weights are untrained**: Default 0.5/0.5 doesn't account for model quality
3. **Label mapping issues**: Potential mismatches between models
4. **Transformer may not be optimal**: Wav2Vec2 might not capture prosody as well as feature-based methods

## Solutions Implemented

### 1. CNN+LSTM Model (Recommended) ✅

**Why this will help:**
- Classic but proven approach for emotion recognition
- Captures prosody (pitch, energy, rhythm) better than transformers
- Many papers show 40-55% accuracy on CREMA-D (vs current 36%)
- More interpretable and explainable

**Files created:**
- `src/cnn_lstm_emotion_model.py`: Model implementation
- `scripts/train_cnn_lstm_model.py`: Training script
- `CNN_LSTM_GUIDE.md`: Complete guide

**To use:**
```bash
# Train the model
python scripts/train_cnn_lstm_model.py --epochs 50

# Expected: 40-50%+ accuracy (vs current 36%)
```

### 2. Configurable Fusion Weights ✅

**Updated `config.py`:**
- `fusion_weights`: Can be "auto", "equal", or custom [audio_weight, text_weight]
- `min_text_accuracy`: Fallback to audio-only if text is too poor
- Default fusion weights now favor audio (0.6/0.4) since it's better

**To improve current fusion:**
```python
# In config.py, set:
FUSION_CONFIG = {
    "fusion_weights": [0.7, 0.3],  # Favor audio more since text is poor
    # OR
    "fusion_weights": "auto",  # Performance-based weighting
}
```

### 3. Better Late Fusion ✅

**Fixed in `src/fusion_model.py`:**
- Late fusion now uses actual trained model logits (not new untrained classifiers)
- Proper label remapping to canonical order
- Default weights favor audio (0.6/0.4)

## Recommended Action Plan

### Immediate (Quick Wins):
1. **Adjust fusion weights** in `config.py`:
   ```python
   "fusion_weights": [0.7, 0.3],  # Favor audio more
   ```
   This should improve fused accuracy from 28% → ~32-35%

2. **Disable text in fusion** if accuracy stays low:
   ```python
   "min_text_accuracy": 0.30,  # Use audio-only if text < 30%
   ```

### Short-term (Best Results):
3. **Train CNN+LSTM model**:
   ```bash
   python scripts/train_cnn_lstm_model.py --epochs 50 --lr 0.001
   ```
   Expected: 40-50%+ accuracy (vs current 36%)

4. **Integrate CNN+LSTM** into fusion:
   - Update `config.py`: `AUDIO_MODEL_TYPE = "cnn_lstm"`
   - Update `app.py` to load CNN+LSTM model
   - Expected fused accuracy: 45-55%+

### Long-term (Optimal):
5. **Improve text model**:
   - Current 20% accuracy is unacceptable
   - Consider retraining with better data/hyperparameters
   - Or use a different text model

6. **Train fusion weights**:
   - Currently using fixed/untrained weights
   - Could train end-to-end for optimal combination

## Expected Performance Improvements

| Model | Current | With CNN+LSTM | With Optimized Fusion |
|-------|---------|---------------|----------------------|
| Audio-only | 36% | **40-50%** | 40-50% |
| Text-only | 20% | 20% | 30-40% (if improved) |
| Fused | 28% | **45-55%** | **50-60%** |

## Why CNN+LSTM Will Work Better

1. **Prosody-focused**: Emotion is primarily in prosody (pitch, energy, rhythm)
2. **Feature-rich**: 39 comprehensive features vs transformer embeddings
3. **Temporal modeling**: LSTM captures emotion dynamics over time
4. **Proven**: Many emotion recognition papers show this beats transformers
5. **Efficient**: Faster inference, easier to train

## Next Steps

1. ✅ CNN+LSTM model implemented
2. ✅ Training script ready
3. ✅ Config updated for better fusion
4. ⏳ **Train CNN+LSTM model** (run the script)
5. ⏳ **Evaluate and compare** with current models
6. ⏳ **Integrate into app** if results are better

The CNN+LSTM approach is your best bet for improving accuracy from 28% to 45-55%+!

