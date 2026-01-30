"""
Test audio emotion inference on sample audio files
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
import librosa

import config
from src.audio_emotion_model import AudioEmotionConfig, DistilHuBERTEmotionModel, AudioEmotionInference
from src.audio_features import extract_prosodic_features
from src.preprocessor import AudioPreprocessor


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model_for_inference(checkpoint_path: Path, device: torch.device):
    """Load model and create inference wrapper"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    cfg_dict = checkpoint["config"]
    cfg = AudioEmotionConfig(**cfg_dict)
    
    model = DistilHuBERTEmotionModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    id2label = checkpoint["id2label"]
    
    inference = AudioEmotionInference(model, id2label, device=device)
    return inference


def process_audio_file(audio_path: Path, preprocessor: AudioPreprocessor):
    """Load and preprocess audio file"""
    audio, sr = preprocessor.load_audio(str(audio_path))
    
    # Extract prosodic features
    prosodic_feats = extract_prosodic_features(audio, sr)
    prosodic_vector = prosodic_feats["prosodic_feature_vector"]
    
    # Convert to tensor
    waveform = torch.tensor(audio, dtype=torch.float32)
    prosodic_tensor = torch.tensor(prosodic_vector, dtype=torch.float32).unsqueeze(0)
    
    return waveform.unsqueeze(0), prosodic_tensor


def predict_emotion(inference: AudioEmotionInference, waveform: torch.Tensor, prosodic: torch.Tensor):
    """Run emotion prediction"""
    results = inference.predict(waveform, prosodic)
    return results


def main():
    parser = argparse.ArgumentParser(description="Test audio emotion inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.PROCESSED_DATA_DIR / "audio_emotion_model" / "best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file to test",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="If audio is a directory, test on N random samples",
    )
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    inference = load_model_for_inference(checkpoint_path, device)
    preprocessor = AudioPreprocessor()
    
    # Process audio file(s)
    audio_path = Path(args.audio)
    
    if audio_path.is_file():
        # Single file
        print(f"\nProcessing: {audio_path.name}")
        waveform, prosodic = process_audio_file(audio_path, preprocessor)
        results = predict_emotion(inference, waveform, prosodic)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Predicted Emotion: {results['pred_labels'][0]}")
        print(f"\nProbabilities:")
        probs = results['probs'][0].cpu().numpy()
        for i, label in inference.id2label.items():
            print(f"  {label:12s}: {probs[i]:.4f}")
        print("=" * 60)
        
    elif audio_path.is_dir():
        # Directory - test on random samples
        import random
        audio_files = list(audio_path.glob("*.wav"))
        if len(audio_files) == 0:
            print(f"No WAV files found in {audio_path}")
            return
        
        sample_files = random.sample(audio_files, min(args.num_samples, len(audio_files)))
        
        print(f"\nTesting on {len(sample_files)} random samples from {audio_path}")
        print("=" * 60)
        
        for audio_file in sample_files:
            print(f"\nFile: {audio_file.name}")
            waveform, prosodic = process_audio_file(audio_file, preprocessor)
            results = predict_emotion(inference, waveform, prosodic)
            
            print(f"  Predicted: {results['pred_labels'][0]}")
            probs = results['probs'][0].cpu().numpy()
            top_3 = sorted(
                [(inference.id2label[i], probs[i]) for i in range(len(inference.id2label))],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            print(f"  Top 3: {', '.join([f'{l}({p:.3f})' for l, p in top_3])}")
        
        print("\n" + "=" * 60)
    else:
        raise ValueError(f"Invalid audio path: {audio_path}")


if __name__ == "__main__":
    main()

