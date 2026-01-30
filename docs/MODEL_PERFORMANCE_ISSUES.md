# Model Performance Issues & Solutions

## Current Problem

The text emotion model is showing poor performance:

### Test Results
- **All predictions are biased** (either "HAPPY" or "DISGUST" for everything)
- **Macro-F1: 0.1671** (16.71%) - barely better than random (16.67% for 6 classes)
- **Validation Accuracy: 44.36%** - better than random but still poor

### Root Causes

1. **Insufficient Training**
   - Only trained for 2 epochs
   - Early stopping triggered too early
   - Model hasn't learned meaningful patterns

2. **Class Imbalance**
   - GoEmotions dataset has severe class imbalance
   - Model is biased toward majority classes (happy, neutral)
   - Minority classes (disgust, fear) are underperforming

3. **Checkpoint Loading Warnings**
   - 201 missing keys, 253 unexpected keys when loading
   - PEFT/LoRA structure might have slight mismatches
   - However, this is likely not the main issue

## Solutions

### Immediate Fix: Retrain with Better Configuration

1. **Use Weighted Loss**
   ```python
   # Calculate class weights from training data
   from sklearn.utils.class_weight import compute_class_weight
   class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
   criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
   ```

2. **Train Longer**
   - Increase `--epochs` from 3 to 5-10
   - Increase `--patience` from 2 to 3-4
   - Allow model more time to learn

3. **Better Hyperparameters**
   - Consider adjusting learning rate (try 1e-5 or 3e-5)
   - Increase LoRA rank (try r=16 instead of r=8)
   - Adjust batch size if memory allows

### Long-term Improvements

1. **Data Augmentation**
   - Balance the dataset by oversampling minority classes
   - Use text augmentation techniques

2. **Per-Class Evaluation**
   - Track F1 score per emotion class
   - Identify which emotions need more training data

3. **Model Architecture**
   - Consider unfreezing base model (Mode B) if LoRA-only isn't sufficient
   - Try different base models (roberta-large, distilroberta)

## Quick Test Command

To verify if the model is working after retraining:

```bash
python scripts/diagnose_model.py
```

This will test the model on a set of known sentences and show predictions.

## Expected Behavior After Fix

- Model should correctly identify emotions in test sentences
- Predictions should vary based on input (not always same class)
- Macro-F1 should be > 0.30 (30%) at minimum
- Per-class F1 scores should be more balanced

