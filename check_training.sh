#!/bin/bash
# Quick script to check training progress

LOG_FILE="/Users/guest1/Desktop/NLP/Speech-to-Text-Sentiment-Analysis-Analyzing-Emotions-in-Spoken-Language/training_mps.log"

if [ -f "$LOG_FILE" ]; then
    echo "=== Training Progress ==="
    tail -15 "$LOG_FILE"
    echo ""
    echo "=== Process Status ==="
    ps aux | grep "train_audio_emotion_improved" | grep -v grep || echo "Training process not found"
else
    echo "Log file not found yet"
fi
