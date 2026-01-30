# Multimodal Emotion Analysis Web UI

A sleek, modern web interface for analyzing emotions from audio using speech-to-text and audio emotion recognition models.

## Features

- 🎧 **Audio Input**
  - Upload audio files (WAV, MP3, FLAC)
  - Live microphone recording
  - Real-time waveform visualization

- 🧠 **Analysis Pipeline**
  - **Whisper STT**: Automatic speech transcription
  - **Audio Emotion Model**: Emotion recognition from raw audio
  - **Text Emotion Model**: Placeholder for future text-based analysis

- 📊 **Visualization**
  - Emoji-based emotion indicators
  - Confidence scores with progress bars
  - Dominant emotion highlighting
  - Smooth animations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have a trained model checkpoint:
   - `data/processed/audio_emotion_model_improved/best_model.pt` (preferred)
   - OR `data/processed/audio_emotion_model/best_model.pt`

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Upload Audio File
1. Click the upload area or drag and drop an audio file
2. Supported formats: WAV, MP3, FLAC
3. The system will automatically:
   - Transcribe the audio using Whisper
   - Analyze emotions from the audio signal
   - Display results with emoji indicators

### Live Recording
1. Click "Start Recording" button
2. Speak into your microphone
3. Click "Stop Recording" when done
4. The analysis will start automatically

### Results Display
- **Transcription**: Shows the transcribed text
- **Audio Emotion**: Displays emotion probabilities from audio analysis
- **Text Emotion**: Placeholder for future text-based emotion analysis

## API Endpoints

### `POST /api/analyze`
Analyzes audio file and returns transcription + emotions.

**Request:**
- `audio`: Audio file (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "transcription": "Transcribed text...",
  "waveform": [...],
  "audio_emotion": {
    "dominant": {
      "label": "happy",
      "emoji": "😊",
      "confidence": 0.85
    },
    "all": {
      "anger": {"probability": 0.05, "emoji": "😠"},
      ...
    }
  },
  "text_emotion": {
    "dominant": {...},
    "all": {...},
    "note": "Text Emotion Model (Coming Soon)"
  }
}
```

### `GET /api/health`
Health check endpoint.

## Technical Details

- **Backend**: Flask with CORS support
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Models**:
  - Whisper (base) for transcription
  - DistilHuBERT-based audio emotion model
- **Device**: Automatically uses MPS (Apple Silicon), CUDA, or CPU

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Requires microphone access for recording feature

## Troubleshooting

### Model Not Found
Ensure you have a trained model checkpoint in one of these locations:
- `data/processed/audio_emotion_model_improved/best_model.pt`
- `data/processed/audio_emotion_model/best_model.pt`

### Microphone Access Denied
- Check browser permissions
- Ensure HTTPS or localhost (required for microphone access)

### Slow Processing
- First request may be slow (model loading)
- Subsequent requests are faster
- Consider using a smaller Whisper model for faster transcription

## Future Enhancements

- [ ] Text-based emotion analysis model
- [ ] Multi-modal emotion fusion
- [ ] Batch processing
- [ ] Export results
- [ ] Audio playback controls
- [ ] Real-time emotion tracking during recording

