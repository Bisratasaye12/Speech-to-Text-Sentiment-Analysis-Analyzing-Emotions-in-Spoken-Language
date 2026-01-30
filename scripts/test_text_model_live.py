#!/usr/bin/env python3
"""
Minimal interface to test text emotion model by recording audio.

Usage:
    python scripts/test_text_model_live.py
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import config
from src.transcriber import WhisperTranscriber
from src.text_model import load_roberta_text_model, TextEmotionPredictor


def record_audio(duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from microphone."""
    print(f"\n🎤 Recording for {duration} seconds... (speak now)")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait until recording is finished
    print("✓ Recording complete")
    return audio.squeeze()


def load_trained_model(device: torch.device, checkpoint_name: str = None) -> TextEmotionPredictor:
    """Load the trained RoBERTa model from checkpoint."""
    checkpoint_dir = config.PROCESSED_DATA_DIR / "models" / "checkpoints"
    
    # Prefer the new 5-epoch model if available, otherwise use default
    if checkpoint_name:
        best_model_path = checkpoint_dir / checkpoint_name
    else:
        # Check for new 5-epoch model first
        new_model_path = checkpoint_dir / "best_model_5epochs.pt"
        default_model_path = checkpoint_dir / "best_model.pt"
        
        if new_model_path.exists():
            best_model_path = new_model_path
            print("📦 Using new 5-epoch trained model")
        elif default_model_path.exists():
            best_model_path = default_model_path
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_dir}\n"
                "Please train the model first or specify a checkpoint file."
            )
    
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {best_model_path}\n"
            "Please train the model first using: python scripts/train_text_roberta.py"
        )
    
    print(f"📦 Loading model from: {best_model_path.relative_to(config.PROJECT_ROOT)}")
    # Load checkpoint on CPU first to avoid MPS alignment issues
    checkpoint = torch.load(best_model_path, map_location="cpu")
    
    # Display checkpoint info
    epoch = checkpoint.get("epoch", "N/A")
    macro_f1 = checkpoint.get("best_macro_f1", 0.0)
    val_acc = checkpoint.get("val_acc", 0.0)
    print(f"   Epoch: {epoch}, Macro-F1: {macro_f1:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load model architecture (must match training config: LoRA + frozen base)
    artifacts = load_roberta_text_model(
        model_name="roberta-base",
        use_lora=True,
        freeze_base=True,
    )
    
    # Load trained weights on CPU first
    # Note: PEFT models store weights in a specific format, but load_state_dict should handle it
    missing_keys, unexpected_keys = artifacts.model.load_state_dict(
        checkpoint["model_state_dict"], 
        strict=False
    )
    if missing_keys:
        print(f"⚠️  Missing keys (expected for PEFT): {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)} keys")
    
    # Move model to target device after loading weights
    artifacts.model.to(device)
    artifacts.model.eval()
    
    print("✓ Model loaded")
    return TextEmotionPredictor(artifacts)


def main():
    print("=" * 60)
    print("Text Emotion Model - Live Audio Test")
    print("=" * 60)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🔧 Using device: {device}")
    
    # Load models
    print("\n[1/4] Loading models...")
    transcriber = WhisperTranscriber(model_name="base")  # Use smaller model for speed
    text_predictor = load_trained_model(device)
    
    # Main loop
    print("\n" + "=" * 60)
    print("Ready! Press Enter to record, or 'q' to quit")
    print("=" * 60)
    
    while True:
        user_input = input("\nPress Enter to record (or 'q' to quit): ").strip().lower()
        if user_input == 'q':
            print("👋 Goodbye!")
            break
        
        try:
            # Record audio
            audio = record_audio(duration=3.0)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio, 16000)
            
            # Transcribe
            print("\n[2/4] Transcribing audio...")
            result = transcriber.transcribe(tmp_path, language="en", verbose=False)
            transcription = result['text'].strip()
            
            if not transcription:
                print("⚠️  No speech detected. Try again.")
                Path(tmp_path).unlink()
                continue
            
            print(f"📝 Transcription: \"{transcription}\"")
            
            # Predict emotion
            print("\n[3/4] Predicting emotion...")
            prediction = text_predictor.predict_text(
                transcription, 
                device,
                return_attention=True,
                return_hidden_states=True
            )
            
            # Display results
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"🎭 Predicted Emotion: {prediction['label'].upper()}")
            print(f"\n📊 Probabilities:")
            sorted_probs = sorted(
                prediction['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for emotion, prob in sorted_probs:
                bar_length = int(prob * 40)
                bar = "█" * bar_length
                print(f"  {emotion:12s} {prob:.3f} {bar}")
            
            # Display attention patterns
            if 'attention_weights' in prediction:
                attn = prediction['attention_weights']
                print(f"\n🔍 Top Words Model Focused On:")
                # Normalize attention weights for better visualization
                top_tokens = attn['top_attended_tokens'][:10]
                if top_tokens:
                    max_weight = max(w for _, w in top_tokens)
                    for token, weight in top_tokens:
                        # Clean up token display (remove Ġ prefix which is RoBERTa's space marker)
                        display_token = token.replace('Ġ', ' ').strip()
                        if display_token and display_token not in ['<s>', '</s>', '<pad>', '']:
                            # Normalize to percentage
                            pct = (weight / max_weight * 100) if max_weight > 0 else 0
                            bar_length = int(pct / 2)  # Scale for display
                            bar = "█" * bar_length
                            print(f"  '{display_token:20s}' {weight:.4f} ({pct:.1f}%) {bar}")
            
            # Display hidden state info
            if 'hidden_states' in prediction:
                hs = prediction['hidden_states']
                stats = hs['hidden_state_stats']
                print(f"\n🧠 Context Vector (Hidden State):")
                print(f"  Dimensions: 768")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Std: {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            
            print("=" * 60)
            
            # Cleanup
            Path(tmp_path).unlink()
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

