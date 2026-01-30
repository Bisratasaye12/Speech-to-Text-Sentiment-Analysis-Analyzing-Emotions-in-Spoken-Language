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
from src.audio_features import extract_prosodic_features, prosodic_feature_vector
from src.preprocessor import AudioPreprocessor

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# Global model instances (lazy loading)
whisper_transcriber = None
audio_emotion_model = None
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
    """Load audio emotion model"""
    global audio_emotion_model
    
    if audio_emotion_model is None:
        device = get_device()
        print(f"Loading audio emotion model on {device}...")
        
        # Try to load from improved model first, fallback to original
        checkpoint_paths = [
            config.PROCESSED_DATA_DIR / "audio_emotion_model_improved" / "best_model.pt",
            config.PROCESSED_DATA_DIR / "audio_emotion_model" / "best_model.pt",
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if path.exists():
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError("No trained model checkpoint found")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg_dict = checkpoint["config"]
        cfg = AudioEmotionConfig(**cfg_dict)
        
        model = DistilHuBERTEmotionModel(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        id2label = checkpoint["id2label"]
        audio_emotion_model = AudioEmotionInference(model, id2label, device=str(device))
        print("✓ Audio emotion model loaded")
    
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
            
            # 2. Audio Emotion Analysis
            emotion_model = load_audio_emotion_model()
            
            # Extract prosodic features (same as training)
            # Note: extract_prosodic_features returns a dict, we need to convert to vector
            prosodic_dict = extract_prosodic_features(audio, sr)
            # Check if it's already a vector or needs conversion
            if "prosodic_feature_vector" in prosodic_dict:
                prosodic_vec = prosodic_dict["prosodic_feature_vector"]
            else:
                prosodic_vec = prosodic_feature_vector(prosodic_dict)
            
            # Convert to tensors (ensure float32 for MPS compatibility)
            waveform_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            prosodic_tensor = torch.tensor(prosodic_vec, dtype=torch.float32).unsqueeze(0)
            
            # Debug: Check tensor shapes and values
            print(f"[DEBUG] Waveform shape: {waveform_tensor.shape}, range: [{waveform_tensor.min():.4f}, {waveform_tensor.max():.4f}]")
            print(f"[DEBUG] Prosodic shape: {prosodic_tensor.shape}, range: [{prosodic_tensor.min():.4f}, {prosodic_tensor.max():.4f}]")
            
            # Predict emotions
            results = emotion_model.predict(waveform_tensor, prosodic_tensor)
            probs = results['probs'][0].cpu().numpy()
            pred_label = results['pred_labels'][0]
            
            # Debug: Print raw probabilities
            print(f"\n[DEBUG] Model id2label: {emotion_model.id2label}")
            print(f"[DEBUG] Raw probabilities: {probs}")
            print(f"[DEBUG] Predicted label: {pred_label}")
            
            # Format emotion probabilities using model's id2label mapping
            audio_emotions = {}
            for emotion_id, label in emotion_model.id2label.items():
                if label in EMOTION_EMOJIS:  # Only include known emotions
                    prob_value = float(probs[emotion_id])
                    audio_emotions[label] = {
                        'probability': prob_value,
                        'emoji': EMOTION_EMOJIS[label]
                    }
                    print(f"[DEBUG] {label} (id={emotion_id}): {prob_value:.4f}")
            
            # 3. Text Emotion (Placeholder)
            text_emotions = {
                'anger': {'probability': 0.15, 'emoji': '😠'},
                'disgust': {'probability': 0.10, 'emoji': '🤢'},
                'fear': {'probability': 0.12, 'emoji': '😨'},
                'happy': {'probability': 0.25, 'emoji': '😊'},
                'neutral': {'probability': 0.20, 'emoji': '😐'},
                'sad': {'probability': 0.18, 'emoji': '😢'}
            }
            
            # Find dominant emotions
            audio_dominant = max(audio_emotions.items(), key=lambda x: x[1]['probability'])
            text_dominant = max(text_emotions.items(), key=lambda x: x[1]['probability'])
            
            response = {
                'success': True,
                'transcription': transcription_text,
                'waveform': waveform_data[:1000],  # Sample for visualization
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
                        'label': text_dominant[0],
                        'emoji': text_dominant[1]['emoji'],
                        'confidence': text_dominant[1]['probability']
                    },
                    'all': text_emotions,
                    'note': 'Text Emotion Model (Coming Soon)'
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
    
    parser = argparse.ArgumentParser(description='Multimodal Emotion Analysis System')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on (default: 5001)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multimodal Emotion Analysis System")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"\nStarting server on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=True, threaded=True)

