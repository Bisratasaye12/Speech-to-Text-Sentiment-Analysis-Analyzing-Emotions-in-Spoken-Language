# RoBERTa Fine-Tuning for Text Emotion Classification

## Overview

This document describes the fine-tuning of RoBERTa for 6-way text emotion classification using the GoEmotions dataset. The fine-tuning approach uses **LoRA (Low-Rank Adaptation)** to efficiently adapt the pre-trained RoBERTa model to our emotion classification task while keeping the base model weights frozen.

## Motivation

The CREMA-D audio dataset has limited text diversity (only ~12 unique sentences), which could lead to overfitting when training text-based emotion classifiers. To address this, we:

1. **Used an external dataset**: GoEmotions (large-scale Reddit comments with emotion labels)
2. **Mapped labels**: Converted GoEmotions' fine-grained labels to our 6 canonical emotions
3. **Fine-tuned RoBERTa**: Pre-trained the model on diverse text before applying it to CREMA-D transcriptions

## Dataset: GoEmotions

### Source
- **Dataset**: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- **Subset**: `raw` (standard GoEmotions subset)
- **Size**: ~58,000 examples (after filtering)
- **Format**: Reddit comments with multi-label emotion annotations

### Label Mapping Strategy

GoEmotions uses 27 fine-grained emotion labels. We map these to our 6 canonical emotions:

| Canonical Emotion | GoEmotions Labels |
|-------------------|-------------------|
| **anger** | anger, annoyance, disapproval |
| **disgust** | disgust |
| **fear** | fear, nervousness |
| **happy** | joy, amusement, excitement, gratitude, love, optimism, relief, pride, admiration |
| **sad** | sadness, disappointment, embarrassment, grief, remorse |
| **neutral** | neutral |

**Mapping Rules**:
- Only examples with **exactly one** canonical emotion are kept (conservative filtering)
- Examples with 0 or >1 canonical emotions are excluded to avoid ambiguous supervision

### Data Preparation

The prepared dataset is saved to:
```
data/raw/goemotions_6class_train_full.csv
```

Columns:
- `text`: The Reddit comment text
- `label`: The canonical emotion label (anger, disgust, fear, happy, neutral, sad)

## Model Architecture

### Base Model
- **Model**: `roberta-base` (125M parameters)
- **Task**: 6-way sequence classification
- **Classification Head**: 2-layer MLP (dense → dropout → output projection)

### LoRA Configuration

**Mode A (Default - Used in Training)**:
- **LoRA enabled**: Yes
- **Base model**: Frozen (weights not updated)
- **Trainable parameters**: Only LoRA adapters + classification head

**LoRA Hyperparameters**:
```python
LoraConfig(
    r=8,                    # Rank of LoRA matrices
    lora_alpha=16,          # Scaling factor
    target_modules=["query", "value"],  # Apply LoRA to attention layers
    lora_dropout=0.1,       # Dropout in LoRA layers
    bias="none",            # Don't train bias terms
    task_type="SEQ_CLS",    # Sequence classification task
)
```

**Why LoRA?**
- **Efficiency**: Only ~0.1% of parameters are trainable (vs. full fine-tuning)
- **Memory**: Lower memory footprint, faster training
- **Stability**: Freezing base model prevents catastrophic forgetting
- **Flexibility**: Can easily switch to Mode B (unfreeze base) if needed

## Training Configuration

### Data Splits
- **Train/Val Split**: Stratified 90/10 split (preserves class distribution)
- **Test Set**: Reserved for final evaluation (not used during training)

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 32 | Number of examples per batch |
| `learning_rate` | 2e-5 | AdamW learning rate |
| `weight_decay` | 0.01 | L2 regularization |
| `warmup_ratio` | 0.1 | Linear warmup for 10% of training steps |
| `max_length` | 128 | Maximum token sequence length |
| `epochs` | 3 | Maximum training epochs |
| `patience` | 2 | Early stopping patience (epochs) |

### Loss Function
- **Loss**: CrossEntropyLoss (standard, not weighted)
- **Note**: Class imbalance is addressed via **macro-F1** metric for early stopping

### Evaluation Metrics

1. **Accuracy**: Overall classification accuracy
2. **Macro-F1**: Average F1 score across all classes (critical for imbalanced data)
   - Used for **early stopping** (not accuracy)
   - Prevents model from collapsing to majority class

### Training Features

- ✅ **Progress bars**: Real-time training/evaluation progress with `tqdm`
- ✅ **Checkpointing**: Saves model after each epoch
- ✅ **Resume capability**: Can resume from any checkpoint
- ✅ **Best model tracking**: Automatically saves best model based on macro-F1
- ✅ **GPU acceleration**: Uses MPS (Metal Performance Shaders) on Apple Silicon

## Training Results

### Performance Metrics

**Best Model (Epoch 2)**:
- **Validation Accuracy**: 44.36%
- **Validation Macro-F1**: 0.1671 (16.71%)
- **Train Loss**: 1.3860
- **Validation Loss**: 1.3678

### Analysis

**Strengths**:
- Accuracy (44.36%) is **better than random** (~16.67% for 6 classes)
- Model is learning meaningful patterns

**Areas for Improvement**:
- **Low macro-F1** (0.1671) indicates **class imbalance issues**
  - Model likely favors majority classes
  - Minority emotions (disgust, fear) may be underperforming
- Training only completed 2 epochs (may need more training)

### Recommendations

1. **Use weighted loss**: Apply class weights to CrossEntropyLoss based on class frequencies
2. **Train longer**: Increase epochs or remove early stopping temporarily
3. **Per-class analysis**: Evaluate F1 scores per emotion to identify weak classes
4. **Data augmentation**: Consider techniques to balance class distribution
5. **Hyperparameter tuning**: Experiment with learning rate, batch size, LoRA rank

## Usage

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure GoEmotions dataset is prepared
python scripts/prepare_goemotions.py --split train+validation
```

### Training

**Basic training**:
```bash
python scripts/train_text_roberta.py \
    --csv-path data/raw/goemotions_6class_train_full.csv \
    --batch-size 32 \
    --epochs 3
```

**With custom checkpoint directory**:
```bash
python scripts/train_text_roberta.py \
    --csv-path data/raw/goemotions_6class_train_full.csv \
    --checkpoint-dir data/processed/models/checkpoints \
    --save-every 1
```

**Resume from checkpoint**:
```bash
python scripts/train_text_roberta.py \
    --csv-path data/raw/goemotions_6class_train_full.csv \
    --resume data/processed/models/checkpoints/checkpoint_epoch_2.pt
```

### Check Training Results

```bash
python scripts/check_training_results.py
```

### Command-Line Arguments

| Argument | Default | Description |
|---------|---------|-------------|
| `--csv-path` | Required | Path to GoEmotions CSV file |
| `--batch-size` | 32 | Training batch size |
| `--learning-rate` | 2e-5 | Learning rate |
| `--weight-decay` | 0.01 | Weight decay |
| `--epochs` | 3 | Number of training epochs |
| `--warmup-ratio` | 0.1 | Warmup ratio |
| `--max-length` | 128 | Max sequence length |
| `--patience` | 2 | Early stopping patience |
| `--checkpoint-dir` | `data/processed/models/checkpoints` | Checkpoint directory |
| `--resume` | None | Path to checkpoint to resume from |
| `--save-every` | 1 | Save checkpoint every N epochs |

## File Structure

### Key Files

```
src/
├── text_model.py              # RoBERTa model with LoRA
├── text_dataset.py            # Dataset loading and splitting
└── text_emotion_mapping.py    # GoEmotions → canonical label mapping

scripts/
├── prepare_goemotions.py      # Download and prepare GoEmotions dataset
├── train_text_roberta.py      # Main training script
└── check_training_results.py  # Check training results from checkpoints

data/
├── raw/
│   └── goemotions_6class_train_full.csv  # Prepared GoEmotions dataset
└── processed/
    └── models/
        └── checkpoints/
            ├── best_model.pt            # Best model checkpoint
            └── checkpoint_epoch_*.pt     # Epoch checkpoints
```

### Model Checkpoints

Checkpoints contain:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: Learning rate scheduler state
- `epoch`: Current epoch number
- `best_macro_f1`: Best macro-F1 score achieved
- `epochs_without_improvement`: Early stopping counter
- `label2id` / `id2label`: Label encoding mappings

## Implementation Details

### Separation of Concerns

The codebase follows a modular design:

1. **`text_emotion_mapping.py`**: Pure label mapping logic (no model/dataset dependencies)
2. **`text_dataset.py`**: Dataset loading, tokenization, stratified splitting
3. **`text_model.py`**: Model architecture (RoBERTa + LoRA + classification head)
4. **`train_text_roberta.py`**: Training orchestration (loops, checkpointing, metrics)

### Device Handling

The training script automatically detects and uses:
1. **CUDA** (if available) - NVIDIA GPUs
2. **MPS** (if available) - Apple Silicon GPUs
3. **CPU** (fallback) - If no GPU available

### Warning Suppression

The following warnings are suppressed (expected behavior):
- "Some weights were not initialized" - Normal when adding new classification head
- "You should probably TRAIN this model" - Informational message
- Multiprocessing semaphore warnings - Harmless macOS warnings

## Future Improvements

### Short-term
- [ ] Add weighted CrossEntropyLoss based on class frequencies
- [ ] Implement per-class F1 score evaluation
- [ ] Add confusion matrix visualization
- [ ] Test set evaluation script

### Medium-term
- [ ] Experiment with Mode B (unfreeze base model)
- [ ] Hyperparameter tuning (learning rate, LoRA rank, etc.)
- [ ] Data augmentation techniques
- [ ] Ensemble methods

### Long-term
- [ ] Transfer learning to CREMA-D transcriptions
- [ ] Multi-modal fusion (audio + text)
- [ ] Real-time inference API
- [ ] Model deployment pipeline

## References

- **RoBERTa**: [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **GoEmotions**: [Demszky et al., 2020](https://arxiv.org/abs/2005.00547)
- **PEFT Library**: [Hugging Face PEFT](https://github.com/huggingface/peft)

## License

This fine-tuning implementation follows the licenses of:
- RoBERTa: MIT License
- GoEmotions: Apache 2.0 License
- PEFT: Apache 2.0 License

