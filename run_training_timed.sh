#!/bin/bash
# Training script with timing

cd "/Users/guest1/Desktop/NLP/Speech-to-Text-Sentiment-Analysis-Analyzing-Emotions-in-Spoken-Language"

echo "=========================================="
echo "Training started at: $(date)"
echo "=========================================="
START_TIME=$(date +%s)

# Run training with optimizations (AMP + memory cleanup) and log output
python3 scripts/train_audio_emotion_improved.py --batch_size 16 --use_amp 2>&1 | tee training_mps.log

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="

