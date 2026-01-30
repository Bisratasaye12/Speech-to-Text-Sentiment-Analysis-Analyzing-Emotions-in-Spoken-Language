"""
Flask API: Accept audio file for processing.
Orchestrates: upload → preprocess → transcribe → (later: text sentiment, audio sentiment, fusion).
"""
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, render_template

import config
from src.preprocessor import AudioPreprocessor
from src.transcriber import WhisperTranscriber
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH_MB * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(config.UPLOAD_FOLDER)

_transcriber = None


def get_transcriber():
    """Return a single shared Whisper transcriber, loading the model on first use."""
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
        _transcriber.load_model()
    return _transcriber


@app.errorhandler(RequestEntityTooLarge)
def handle_413(e):
    return (
        jsonify({
            "success": False,
            "error": "File too large",
            "max_mb": config.MAX_CONTENT_LENGTH_MB,
        }),
        413,
    )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Simple upload page for testing."""
    return render_template("upload.html")


@app.route("/api/process", methods=["POST"])
def process_audio():
    """
    Accept an audio file, preprocess it, transcribe it.
    Returns JSON with transcription and placeholders for text_sentiment, audio_sentiment, fused_emotion.
    """
    if "file" not in request.files and "audio" not in request.files:
        return jsonify({"success": False, "error": "No file part; use 'file' or 'audio' key"}), 400

    file = request.files.get("file") or request.files.get("audio")
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"File type not allowed. Allowed: {list(config.ALLOWED_EXTENSIONS)}",
        }), 400

    # Save upload with unique name to avoid collisions
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    upload_path = Path(app.config["UPLOAD_FOLDER"]) / safe_name
    file.save(str(upload_path))

    preprocessed_path = Path(app.config["UPLOAD_FOLDER"]) / (
        safe_name.rsplit(".", 1)[0] + "_preprocessed.wav"
    )
    try:
        # Preprocess for Whisper (16 kHz mono, normalized)
        preprocessor = AudioPreprocessor()
        _, _, meta = preprocessor.preprocess(
            str(upload_path), str(preprocessed_path), save=True
        )
        duration_seconds = meta.get("duration", 0.0)

        # Transcribe (uses shared transcriber, loaded once)
        transcriber = get_transcriber()
        result = transcriber.transcribe(str(preprocessed_path), language="en")

        # Response shape ready for fusion (text_sentiment, audio_sentiment, fused_emotion = placeholders)
        response = {
            "success": True,
            "filename": file.filename,
            "transcription": result["text"],
            "language": result["language"],
            "audio_duration_seconds": round(duration_seconds, 2),
            "text_sentiment": None,
            "audio_sentiment": None,
            "fused_emotion": None,
        }
        return jsonify(response)

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"success": False, "error": "Processing failed"}), 500
    finally:
        # Optional: remove temp files to save disk
        if upload_path.exists():
            upload_path.unlink(missing_ok=True)
        if preprocessed_path.exists():
            preprocessed_path.unlink(missing_ok=True)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
