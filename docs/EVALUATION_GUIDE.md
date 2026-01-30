# Fusion Model Evaluation Guide

## Quick Start

Evaluate the fused model's accuracy on the test set:

```bash
# Full evaluation on all test samples
python scripts/evaluate_fusion_model.py

# Quick test with limited samples (faster)
python scripts/evaluate_fusion_model.py --max-samples 50

# Use custom test CSV
python scripts/evaluate_fusion_model.py --test-csv data/splits/test.csv
```

## What It Evaluates

The script compares three models:

1. **Audio-Only Model**: Wav2Vec2-base + prosodic features
2. **Text-Only Model**: RoBERTa-base with LoRA (fine-tuned on GoEmotions)
3. **Fused Model**: Late fusion combining audio + text predictions

## Metrics Reported

- **Accuracy**: Overall classification accuracy
- **Macro F1**: Average F1 score across all classes (handles class imbalance)
- **Weighted F1**: F1 score weighted by class frequency
- **Per-Class Performance**: Precision, recall, F1 for each emotion
- **Confusion Matrix**: Shows which emotions are confused with each other
- **Improvement**: How much better fused model is vs audio-only

## Expected Output

```
======================================================================
Loading Models
======================================================================

1. Loading audio emotion model...
   ✓ Audio model loaded: best_model.pt
   Labels: ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']

2. Loading text emotion model...
   Loading from: data/processed/text_model/best_model.pt
   ✓ Text model loaded
   Labels: ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']

3. Loading Whisper transcriber...
   ✓ Whisper loaded

4. Loading fusion model...
   ✓ Fusion model loaded (type: late)

======================================================================
Evaluating on Test Set
======================================================================

Processing: 100%|████████████| 1119/1119 [XX:XX<00:00, X.XXit/s]
✓ Processed 1119 samples
  Text predictions available: 1119/1119

======================================================================
Results
======================================================================

1. Audio-Only Model:
   Accuracy:  0.XXXX (XX.XX%)
   Macro F1:  0.XXXX
   Weighted F1: 0.XXXX

2. Text-Only Model:
   Accuracy:  0.XXXX (XX.XX%)
   Macro F1:  0.XXXX
   Weighted F1: 0.XXXX
   (Evaluated on 1119 samples with text)

3. Fused Model (late fusion):
   Accuracy:  0.XXXX (XX.XX%)
   Macro F1:  0.XXXX
   Weighted F1: 0.XXXX

4. Improvement over Audio-Only:
   Accuracy:  +0.XXXX (+X.XX%)
   Macro F1:  +0.XXXX (+X.XX%)

5. Per-Class Performance (Fused Model):
   [Detailed classification report]

6. Confusion Matrix (Fused Model):
   [Confusion matrix showing prediction patterns]

7. Per-Class Accuracy (Fused Model):
   anger      : 0.XXXX (XX.XX%) - XXX samples
   disgust    : 0.XXXX (XX.XX%) - XXX samples
   ...
```

## Interpretation

### Good Results
- **Fused accuracy > Audio accuracy**: Fusion is helping
- **Macro F1 improvement > 2-3%**: Significant improvement
- **Balanced per-class performance**: All emotions detected well

### Potential Issues
- **Fused accuracy ≈ Audio accuracy**: Fusion not helping (may need training)
- **Large confusion between specific emotions**: Model struggling with similar emotions
- **Low accuracy on specific class**: Class imbalance or insufficient training data

## Tips

1. **Start with limited samples** (`--max-samples 50`) to test quickly
2. **Check confusion matrix** to see which emotions are confused
3. **Compare per-class F1** to identify weak classes
4. **Monitor improvement** - fused should be better than audio-only

## Troubleshooting

### "No trained model checkpoint found"
- Ensure audio model exists at: `data/processed/audio_emotion_model/best_model.pt`
- Ensure text model exists at: `data/processed/text_model/best_model.pt`

### "Audio file not found"
- Check that audio files are processed and exist
- Verify `processed_filepath` column in test CSV is correct

### Slow evaluation
- Use `--max-samples` to limit test size
- Transcription is the slowest step (uses Whisper)

## Next Steps

If fusion accuracy is good:
- ✅ System is working correctly
- Consider training fusion weights on validation set for better performance

If fusion accuracy is poor:
- Check if label mappings are correct (see debug output)
- Verify both models are trained properly
- Consider training the fusion model weights

