"""
Flask Backend API for Multimodal Emotion Analysis System
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import os
import io
import tempfile
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import librosa
import soundfile as sf

import config
from src.transcriber import WhisperTranscriber
from src.audio_emotion_model import AudioEmotionConfig, DistilHuBERTEmotionModel, AudioEmotionInference
from src.cnn_lstm_emotion_model import CNNLSTMConfig, CNNLSTMEmotionModel, CNNLSTMInference
from src.audio_features import extract_prosodic_features, prosodic_feature_vector
from src.preprocessor import AudioPreprocessor
from src.text_model import load_roberta_text_model, TextModelArtifacts
from src.fusion_model import FusionConfig, MultimodalFusionModel, MultimodalEmotionInference

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# Global model instances (lazy loading)
whisper_transcriber = None
audio_emotion_model = None
text_model_artifacts = None
fusion_model = None
multimodal_inference = None
audio_preprocessor = None

# Emotion labels with emojis
EMOTION_EMOJIS = {
    'anger': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢'
}

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']


def get_device():
    """Get the best available device"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_audio_emotion_model():
    """Load audio emotion model (CNN+LSTM or Wav2Vec2 based on config)"""
    global audio_emotion_model
    
    if audio_emotion_model is None:
        device = get_device()
        model_type = config.AUDIO_MODEL_TYPE
        print(f"Loading audio emotion model ({model_type}) on {device}...")
        
        if model_type == "cnn_lstm":
            # Load CNN+LSTM model
            checkpoint_path = config.PROCESSED_DATA_DIR / "cnn_lstm_model" / "best_model.pt"
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"CNN+LSTM model not found at {checkpoint_path}")
            
            print(f"  Loading CNN+LSTM model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            cfg_dict = checkpoint["config"]
            cfg = CNNLSTMConfig(**cfg_dict)
            
            # Print model configuration
            print(f"  Model type: CNN+LSTM")
            print(f"  Feature dim: {cfg.feature_dim}")
            print(f"  CNN channels: {cfg.cnn_channels}")
            print(f"  LSTM hidden: {cfg.lstm_hidden}")
            print(f"  Num labels: {cfg.num_labels}")
            
            model = CNNLSTMEmotionModel(cfg)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            
            id2label = checkpoint["id2label"]
            print(f"  Labels: {[id2label[i] for i in sorted(id2label.keys())]}")
            
            audio_emotion_model = CNNLSTMInference(model, id2label, device=str(device))
            print("✓ CNN+LSTM audio emotion model loaded")
            
        else:
            # Load Wav2Vec2 model (original)
            checkpoint_paths = [
                config.PROCESSED_DATA_DIR / "audio_emotion_model" / "best_model.pt",
                config.PROCESSED_DATA_DIR / "audio_emotion_model_improved" / "best_model.pt",
            ]
            
            checkpoint_path = None
            for path in checkpoint_paths:
                if path.exists():
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError("No trained Wav2Vec2 model checkpoint found")
            
            print(f"  Loading Wav2Vec2 model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            cfg_dict = checkpoint["config"]
            cfg = AudioEmotionConfig(**cfg_dict)
            
            print(f"  Base model: {cfg.base_model_name}")
            print(f"  Hidden dim: {cfg.hidden_dim}")
            print(f"  Num labels: {cfg.num_labels}")
            
            model = DistilHuBERTEmotionModel(cfg)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            
            id2label = checkpoint["id2label"]
            print(f"  Labels: {[id2label[i] for i in sorted(id2label.keys())]}")
            
            audio_emotion_model = AudioEmotionInference(model, id2label, device=str(device))
            print("✓ Wav2Vec2 audio emotion model loaded")
    
    return audio_emotion_model


def load_whisper():
    """Load Whisper model"""
    global whisper_transcriber
    
    if whisper_transcriber is None:
        print("Loading Whisper model...")
        whisper_transcriber = WhisperTranscriber(model_name="base")  # Using base for faster inference
        whisper_transcriber.load_model()
        print("✓ Whisper model loaded")
    
    return whisper_transcriber


def get_preprocessor():
    """Get audio preprocessor"""
    global audio_preprocessor
    
    if audio_preprocessor is None:
        audio_preprocessor = AudioPreprocessor()
    
    return audio_preprocessor


def load_text_emotion_model():
    """Load text emotion model (RoBERTa with LoRA)"""
    global text_model_artifacts
    
    if text_model_artifacts is None:
        device = get_device()
        print(f"Loading text emotion model on {device}...")
        
        # Try to load from checkpoint if available (prioritize trained models)
        checkpoint_paths = [
            config.PROCESSED_DATA_DIR / "models" / "checkpoints" / "best_model_5epochs.pt",  # NEW: Best trained model
            config.PROCESSED_DATA_DIR / "text_model" / "best_model.pt",  # Primary trained model location
            config.PROCESSED_DATA_DIR / "models" / "checkpoints" / "best_model.pt",
            config.PROCESSED_DATA_DIR / "models" / "text_roberta_lora_best.pt",
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if path.exists():
                checkpoint_path = path
                break
        
        # Load model artifacts (on CPU first to avoid MPS issues)
        artifacts = load_roberta_text_model(
            model_name="roberta-base",
            use_lora=True,
            freeze_base=True,
        )
        
        # Load checkpoint if available
        if checkpoint_path is not None:
            print(f"  Loading trained text model from: {checkpoint_path}")
            
            # CRITICAL: Load on CPU first to avoid MPS alignment issues
            # This is a workaround for PyTorch MPS bug: https://github.com/pytorch/pytorch/issues/84930
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Load model state on CPU first
            try:
                # Get state dict
                state_dict = checkpoint["model_state_dict"]
                
                # Load on CPU (model should still be on CPU from initialization)
                missing_keys, unexpected_keys = artifacts.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    print(f"  ⚠️ Missing keys: {len(missing_keys)} (this is normal for LoRA models)")
                if unexpected_keys:
                    print(f"  ⚠️ Unexpected keys: {len(unexpected_keys)}")
                
                # Now move model to device (after loading weights)
                artifacts.model.to(device)
                print("  ✓ Model weights loaded successfully")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not load model weights: {e}")
                print("  Attempting alternative loading method...")
                try:
                    # Alternative: filter and load compatible keys only
                    state_dict = checkpoint["model_state_dict"]
                    model_dict = artifacts.model.state_dict()
                    # Only load keys that exist in both
                    compatible_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(compatible_dict)
                    artifacts.model.load_state_dict(model_dict, strict=False)
                    artifacts.model.to(device)
                    print(f"  ✓ Model weights loaded (alternative method, {len(compatible_dict)} keys)")
                except Exception as e2:
                    print(f"  ⚠️ Warning: Alternative loading also failed: {e2}")
                    print("  Using pre-trained RoBERTa-base only")
                    artifacts.model.to(device)
            
            # Update label mappings if available
            if "id2label" in checkpoint:
                artifacts.id2label = checkpoint["id2label"]
                print(f"  ✓ Label mapping loaded: {list(artifacts.id2label.values())}")
            if "label2id" in checkpoint:
                artifacts.label2id = checkpoint["label2id"]
            
            # Print model info if available
            if "epoch" in checkpoint:
                print(f"  Model trained for {checkpoint['epoch']} epochs")
            if "val_metrics" in checkpoint or "best_macro_f1" in checkpoint:
                metrics = checkpoint.get("val_metrics", {})
                if "macro_f1" in metrics:
                    print(f"  Validation macro-F1: {metrics['macro_f1']:.4f}")
                elif "best_macro_f1" in checkpoint:
                    print(f"  Best validation macro-F1: {checkpoint['best_macro_f1']:.4f}")
        else:
            print("  ⚠️ No trained text model checkpoint found")
            print("  Using pre-trained RoBERTa-base (not fine-tuned for emotions)")
        
        artifacts.model.to(device)
        artifacts.model.eval()
        text_model_artifacts = artifacts
        print("✓ Text emotion model loaded")
    
    return text_model_artifacts


def normalize_label_mapping(id2label, canonical_order=None):
    """
    Normalize label mapping to ensure consistent ordering.
    Returns a remapping function that converts from model's indices to canonical indices.
    """
    if canonical_order is None:
        canonical_order = config.CANONICAL_EMOTIONS
    
    # Create mapping from model's label order to canonical order
    label_to_canonical_idx = {label: idx for idx, label in enumerate(canonical_order)}
    
    # Create remapping: model_idx -> canonical_idx
    remap = {}
    for model_idx, label in id2label.items():
        if label in label_to_canonical_idx:
            remap[model_idx] = label_to_canonical_idx[label]
        else:
            print(f"⚠️ Warning: Label '{label}' not in canonical order")
    
    return remap


def remap_probabilities(probs, remap_dict, num_labels):
    """Remap probabilities from model's label order to canonical order."""
    remapped = np.zeros(num_labels)
    for model_idx, canonical_idx in remap_dict.items():
        if model_idx < len(probs):
            remapped[canonical_idx] = probs[model_idx]
    return remapped


def load_fusion_model():
    """Load or create fusion model"""
    global fusion_model, multimodal_inference
    
    if fusion_model is None:
        device = get_device()
        print(f"Loading fusion model on {device}...")
        
        # Load audio and text models
        audio_model = load_audio_emotion_model()
        text_artifacts = load_text_emotion_model()
        
        # Validate and normalize label mappings
        audio_id2label = audio_model.id2label
        text_id2label = text_artifacts.id2label
        
        print(f"  Audio model labels: {[audio_id2label[i] for i in sorted(audio_id2label.keys())]}")
        print(f"  Text model labels: {[text_id2label[i] for i in sorted(text_id2label.keys())]}")
        
        # Check if label orders match
        audio_labels_ordered = [audio_id2label[i] for i in sorted(audio_id2label.keys())]
        text_labels_ordered = [text_id2label[i] for i in sorted(text_id2label.keys())]
        
        if audio_labels_ordered != text_labels_ordered:
            print(f"  ⚠️ Label order mismatch detected!")
            print(f"     Audio: {audio_labels_ordered}")
            print(f"     Text:  {text_labels_ordered}")
            print(f"     Using canonical order for fusion: {config.CANONICAL_EMOTIONS}")
        
        # Get audio embedding dimension from audio model config
        if config.AUDIO_MODEL_TYPE == "cnn_lstm":
            # CNN+LSTM: embedding is lstm_hidden * 2 (bidirectional)
            audio_emb_dim = audio_model.model.cfg.lstm_hidden * 2
        else:
            # Wav2Vec2: embedding is hidden_dim
            audio_emb_dim = audio_model.model.cfg.hidden_dim
        
        # Create fusion config from config.py settings
        fusion_cfg = FusionConfig(
            audio_embedding_dim=audio_emb_dim,
            text_embedding_dim=config.FUSION_CONFIG["text_embedding_dim"],
            num_labels=len(config.CANONICAL_EMOTIONS),
            fusion_hidden_dim=config.FUSION_CONFIG["fusion_hidden_dim"],
            dropout=config.FUSION_CONFIG["dropout"],
            fusion_type=config.FUSION_CONFIG["fusion_type"],
        )
        
        fusion_type = config.FUSION_CONFIG["fusion_type"]
        print(f"  Fusion strategy: {fusion_type}")
        
        # Create fusion model
        fusion_model = MultimodalFusionModel(fusion_cfg)
        
        # Try to load fusion model checkpoint if available
        fusion_checkpoint_path = config.PROCESSED_DATA_DIR / "models" / "fusion_model.pt"
        if fusion_checkpoint_path.exists():
            print(f"  Loading fusion model checkpoint from {fusion_checkpoint_path}")
            checkpoint = torch.load(fusion_checkpoint_path, map_location=device)
            fusion_model.load_state_dict(checkpoint["model_state_dict"])
            print("  ✓ Fusion model weights loaded from checkpoint")
        else:
            if fusion_type in ["early", "attention", "weighted"]:
                print(f"  ⚠️ No fusion model checkpoint found for '{fusion_type}' fusion")
                print(f"     Using randomly initialized weights - fusion may not work correctly!")
                print(f"     Recommendation: Use 'late' fusion (combines trained model predictions)")
            elif fusion_type == "late":
                print(f"  ℹ️ Using 'late' fusion (combines trained model predictions)")
                print(f"     No checkpoint needed - uses actual model logits")
        
        fusion_model.to(device)
        fusion_model.eval()
        
        # Use canonical order for fusion model (consistent across all models)
        canonical_id2label = {i: label for i, label in enumerate(config.CANONICAL_EMOTIONS)}
        multimodal_inference = MultimodalEmotionInference(
            audio_model=audio_model,
            text_model=text_artifacts.model,
            fusion_model=fusion_model,
            id2label=canonical_id2label,  # Use canonical order
            device=str(device),
            audio_id2label=audio_id2label,  # Pass for remapping
            text_id2label=text_id2label,  # Pass for remapping
        )
        print("✓ Fusion model loaded")
    
    return fusion_model, multimodal_inference


@app.route('/')
def index():
    """Serve the main UI"""
    return send_from_directory('web', 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(get_device())
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """
    Main analysis endpoint
    Accepts audio file and returns transcription + emotions
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_path = tmp_file.name
            audio_file.save(audio_path)
        
        try:
            # Load and preprocess audio (same as training)
            preprocessor = get_preprocessor()
            audio, sr = preprocessor.load_audio(audio_path)
            
            # Ensure audio is normalized (as in training)
            if preprocessor.normalize:
                audio = preprocessor.normalize_audio(audio)
            
            # Get waveform data for visualization
            waveform_data = audio.tolist()
            
            # 1. Whisper Transcription
            whisper = load_whisper()
            transcription_result = whisper.transcribe(audio_path, language="en")
            transcription_text = transcription_result['text']
            
            # 2. Prepare audio tensors for emotion model
            waveform_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            
            # Extract prosodic features only if using Wav2Vec2 (CNN+LSTM extracts internally)
            if config.AUDIO_MODEL_TYPE == "cnn_lstm":
                # CNN+LSTM doesn't need prosodic features (will be ignored)
                prosodic_tensor = torch.zeros(1, 13, dtype=torch.float32)  # Dummy tensor for compatibility
            else:
                # Wav2Vec2 needs prosodic features
                prosodic_dict = extract_prosodic_features(audio, sr)
                if "prosodic_feature_vector" in prosodic_dict:
                    prosodic_vec = prosodic_dict["prosodic_feature_vector"]
                else:
                    prosodic_vec = prosodic_feature_vector(prosodic_dict)
                prosodic_tensor = torch.tensor(prosodic_vec, dtype=torch.float32).unsqueeze(0)
            
            # 3. Multimodal Emotion Analysis (Audio + Text + Fusion)
            fusion_model_instance, multimodal_inf = load_fusion_model()
            text_artifacts = load_text_emotion_model()
            audio_model = load_audio_emotion_model()
            
            # Get label mappings from both models
            audio_id2label = audio_model.id2label  # Audio model's label mapping
            text_id2label = text_artifacts.id2label  # Text model's label mapping
            canonical_id2label = {i: label for i, label in enumerate(config.CANONICAL_EMOTIONS)}
            
            # Create remapping functions to normalize to canonical order
            audio_remap = normalize_label_mapping(audio_id2label)
            text_remap = normalize_label_mapping(text_id2label)
            
            # Run multimodal prediction
            multimodal_results = multimodal_inf.predict(
                waveforms=waveform_tensor,
                prosodic_feats=prosodic_tensor,
                text=transcription_text if transcription_text.strip() else None,
                text_tokenizer=text_artifacts.tokenizer,
                return_individual=True,
            )
            
            # Extract results
            fused_probs = multimodal_results['fused_probs'][0].cpu().numpy()
            fused_pred_label = multimodal_results['fused_pred_labels'][0]
            
            # Audio-only results - remap to canonical order
            audio_probs_raw = multimodal_results['audio_probs'][0].cpu().numpy()
            audio_probs = remap_probabilities(audio_probs_raw, audio_remap, len(config.CANONICAL_EMOTIONS))
            # Recalculate audio prediction from remapped probabilities
            audio_pred_id = np.argmax(audio_probs)
            audio_pred_label = canonical_id2label[audio_pred_id]
            
            # Text-only results (if available) - remap to canonical order
            text_probs = None
            text_pred_label = None
            if 'text_probs' in multimodal_results:
                text_probs_raw = multimodal_results['text_probs'][0].cpu().numpy()
                text_probs = remap_probabilities(text_probs_raw, text_remap, len(config.CANONICAL_EMOTIONS))
                # Recalculate text prediction from remapped probabilities
                text_pred_id = np.argmax(text_probs)
                text_pred_label = canonical_id2label[text_pred_id]
            
            # Format emotion probabilities - all use canonical order now
            def format_emotions(probs, id2label, model_name="unknown"):
                emotions = {}
                # Ensure we iterate in the correct order (0, 1, 2, ...)
                for emotion_id in sorted(id2label.keys()):
                    label = id2label[emotion_id]
                    if label in EMOTION_EMOJIS:
                        # Make sure we don't go out of bounds
                        if emotion_id < len(probs):
                            prob_value = float(probs[emotion_id])
                            emotions[label] = {
                                'probability': prob_value,
                                'emoji': EMOTION_EMOJIS[label]
                            }
                        else:
                            print(f"⚠️ Warning: {model_name} emotion_id {emotion_id} out of range for probs length {len(probs)}")
                return emotions
            
            # All predictions now use canonical order for consistency
            fused_emotions = format_emotions(fused_probs, canonical_id2label, "fused")
            audio_emotions = format_emotions(audio_probs, canonical_id2label, "audio")
            
            text_emotions = {}
            if text_probs is not None:
                text_emotions = format_emotions(text_probs, canonical_id2label, "text")
            else:
                # Fallback if text model didn't produce results
                text_emotions = {label: {'probability': 0.0, 'emoji': emoji} 
                               for label, emoji in EMOTION_EMOJIS.items()}
            
            # Debug: Print top predictions for verification
            print(f"\n[DEBUG] Predictions (canonical order):")
            print(f"  Fused: {fused_pred_label} (confidence: {fused_probs[np.argmax(fused_probs)]:.4f})")
            print(f"    Probs: {dict(zip([canonical_id2label[i] for i in range(len(fused_probs))], [f'{p:.4f}' for p in fused_probs]))}")
            print(f"  Audio: {audio_pred_label} (confidence: {audio_probs[np.argmax(audio_probs)]:.4f})")
            print(f"    Probs: {dict(zip([canonical_id2label[i] for i in range(len(audio_probs))], [f'{p:.4f}' for p in audio_probs]))}")
            if text_probs is not None:
                print(f"  Text:  {text_pred_label} (confidence: {text_probs[np.argmax(text_probs)]:.4f})")
                print(f"    Probs: {dict(zip([canonical_id2label[i] for i in range(len(text_probs))], [f'{p:.4f}' for p in text_probs]))}")
            
            # Check if remapping worked correctly
            print(f"\n[DEBUG] Label mappings:")
            print(f"  Audio model order: {[audio_id2label[i] for i in sorted(audio_id2label.keys())]}")
            print(f"  Text model order:  {[text_id2label[i] for i in sorted(text_id2label.keys())]}")
            print(f"  Canonical order:   {config.CANONICAL_EMOTIONS}")
            print(f"  Audio remap: {audio_remap}")
            print(f"  Text remap:  {text_remap}")
            
            # Find dominant emotions - recalculate from remapped probabilities to ensure correctness
            # This ensures we use the canonical order and remapped probabilities
            fused_dominant = max(fused_emotions.items(), key=lambda x: x[1]['probability'])
            audio_dominant = max(audio_emotions.items(), key=lambda x: x[1]['probability'])
            text_dominant = max(text_emotions.items(), key=lambda x: x[1]['probability']) if text_emotions else None
            
            # Verify predictions match calculated dominants
            fused_pred_id_calc = np.argmax(fused_probs)
            fused_pred_label_calc = canonical_id2label[fused_pred_id_calc]
            
            if fused_pred_label != fused_pred_label_calc:
                print(f"  ⚠️ Warning: Fused model prediction '{fused_pred_label}' != calculated '{fused_pred_label_calc}'")
                print(f"     Using calculated: {fused_pred_label_calc} (confidence: {fused_dominant[1]['probability']:.4f})")
                fused_pred_label = fused_pred_label_calc
            
            if audio_pred_label != audio_dominant[0]:
                print(f"  ⚠️ Warning: Audio prediction '{audio_pred_label}' != calculated '{audio_dominant[0]}'")
                print(f"     Using calculated: {audio_dominant[0]} (confidence: {audio_dominant[1]['probability']:.4f})")
                audio_pred_label = audio_dominant[0]
            
            if text_probs is not None and text_pred_label != text_dominant[0]:
                print(f"  ⚠️ Warning: Text prediction '{text_pred_label}' != calculated '{text_dominant[0]}'")
                print(f"     Using calculated: {text_dominant[0]} (confidence: {text_dominant[1]['probability']:.4f})")
                text_pred_label = text_dominant[0]
            
            # Final verification
            print(f"\n[DEBUG] Final predictions:")
            print(f"  Fused: {fused_pred_label} ({fused_dominant[1]['probability']:.4f})")
            print(f"  Audio: {audio_pred_label} ({audio_dominant[1]['probability']:.4f})")
            if text_dominant:
                print(f"  Text:  {text_pred_label} ({text_dominant[1]['probability']:.4f})")
            
            # Print fusion weights if available
            if 'weights' in multimodal_results:
                weights = multimodal_results['weights']
                print(f"  Fusion weights: audio={weights[0].item():.4f}, text={weights[1].item():.4f}")
            
            response = {
                'success': True,
                'transcription': transcription_text,
                'waveform': waveform_data[:1000],  # Sample for visualization
                'fused_emotion': {
                    'dominant': {
                        'label': fused_dominant[0],
                        'emoji': fused_dominant[1]['emoji'],
                        'confidence': fused_dominant[1]['probability']
                    },
                    'all': fused_emotions,
                    'note': 'Multimodal Fusion (Audio + Text)'
                },
                'audio_emotion': {
                    'dominant': {
                        'label': audio_dominant[0],
                        'emoji': audio_dominant[1]['emoji'],
                        'confidence': audio_dominant[1]['probability']
                    },
                    'all': audio_emotions
                },
                'text_emotion': {
                    'dominant': {
                        'label': text_dominant[0] if text_dominant else 'neutral',
                        'emoji': text_dominant[1]['emoji'] if text_dominant else EMOTION_EMOJIS['neutral'],
                        'confidence': text_dominant[1]['probability'] if text_dominant else 0.0
                    },
                    'all': text_emotions,
                    'note': 'Text-only prediction' if text_probs is not None else 'Text model not available'
                }
            }
            
            return jsonify(response)
        
        finally:
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/transcribe', methods=['POST'])
def transcribe_only():
    """Transcribe audio only (without emotion analysis)"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_path = tmp_file.name
            audio_file.save(audio_path)
        
        try:
            whisper = load_whisper()
            result = whisper.transcribe(audio_path, language="en")
            
            return jsonify({
                'success': True,
                'text': result['text'],
                'language': result['language']
            })
        
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    import argparse
    import ssl
    
    parser = argparse.ArgumentParser(description='Multimodal Emotion Analysis System')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on (default: 5001)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--ssl-cert', type=str, default=None, help='Path to SSL certificate file (for HTTPS)')
    parser.add_argument('--ssl-key', type=str, default=None, help='Path to SSL private key file (for HTTPS)')
    parser.add_argument('--ssl-adhoc', action='store_true', help='Use ad-hoc SSL certificate (for development)')
    args = parser.parse_args()
    
    # Setup SSL context if certificates provided
    ssl_context = None
    use_https = False
    
    if args.ssl_cert and args.ssl_key:
        ssl_context = (args.ssl_cert, args.ssl_key)
        use_https = True
    elif args.ssl_adhoc:
        # Use ad-hoc certificate for development (self-signed)
        ssl_context = 'adhoc'
        use_https = True
    
    protocol = 'https' if use_https else 'http'
    
    print("=" * 60)
    print("Multimodal Emotion Analysis System")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"\nStarting server on {protocol}://localhost:{args.port}")
    if use_https:
        print("✓ HTTPS enabled")
        if args.ssl_adhoc:
            print("⚠ Using self-signed certificate (browser will show security warning)")
    else:
        print("ℹ Running on HTTP (use --ssl-adhoc for HTTPS in development)")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(
        host=args.host, 
        port=args.port, 
        debug=True, 
        threaded=True,
        ssl_context=ssl_context
    )
