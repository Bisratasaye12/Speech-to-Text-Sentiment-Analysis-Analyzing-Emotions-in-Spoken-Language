#!/bin/bash
# Training script for MELD dataset - run in tmux

cd "$(dirname "$0")"

echo "============================================================"
echo "MELD Dataset Training"
echo "============================================================"
echo "Starting training at $(date)"
echo ""

python3 scripts/train_audio_emotion_improved.py \
    --train_csv data/meld/splits/train.csv \
    --val_csv data/meld/splits/val.csv \
    --batch_size 8 \
    --epochs 12 \
    --lr 5e-5 \
    --patience 3 \
    --use_amp \
    --freeze_backbone \
    --init_checkpoint data/processed/audio_emotion_model_improved/best_model.pt \
    --output_dir data/meld/models \
    2>&1 | tee data/meld/training.log

echo ""
echo "Training completed at $(date)"

