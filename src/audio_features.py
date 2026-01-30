"""
Prosodic Feature Extraction Utilities
-------------------------------------

These helpers extract prosodic cues such as:
- Pitch (F0) statistics
- Energy (RMS) statistics
- Spectral / voicing statistics
- A simple proxy for speech rate

All functions operate on mono, 16 kHz audio (as configured in `config.AUDIO_CONFIG`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import librosa


@dataclass
class ProsodicConfig:
    """Configuration for prosodic feature extraction."""

    sr: int = 16000
    frame_length: int = 1024
    hop_length: int = 256
    fmin: float = 50.0
    fmax: float = 500.0


def _safe_stats(x: np.ndarray) -> Tuple[float, float, float]:
    """Return (mean, std, max) with safe handling for empty arrays."""
    if x.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(x)), float(np.std(x)), float(np.max(x))


def extract_prosodic_features(
    audio: np.ndarray,
    sr: int,
    cfg: ProsodicConfig | None = None,
) -> Dict[str, float]:
    """
    Extract a compact set of prosodic features from a waveform.

    Args:
        audio: 1D mono waveform.
        sr: Sample rate of `audio`.
        cfg: Optional `ProsodicConfig`. If None, defaults are used.

    Returns:
        Dictionary of scalar prosodic features.
    """
    if cfg is None:
        cfg = ProsodicConfig(sr=sr)

    # Ensure float32
    y = librosa.util.normalize(audio.astype(np.float32))

    # Pitch using librosa.yin (fundamental frequency)
    try:
        f0 = librosa.yin(
            y,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            sr=sr,
            frame_length=cfg.frame_length,
            hop_length=cfg.hop_length,
        )
        # Ignore unvoiced (nan) frames
        f0_valid = f0[~np.isnan(f0)]
    except Exception:
        f0_valid = np.array([], dtype=np.float32)

    f0_mean, f0_std, f0_max = _safe_stats(f0_valid)

    # Energy (RMS)
    rms = librosa.feature.rms(
        y=y, frame_length=cfg.frame_length, hop_length=cfg.hop_length
    )[0]
    rms_mean, rms_std, rms_max = _safe_stats(rms)

    # Zero-crossing rate (proxy for noisiness / fricatives)
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=cfg.frame_length, hop_length=cfg.hop_length
    )[0]
    zcr_mean, zcr_std, zcr_max = _safe_stats(zcr)

    # Spectral centroid (brightness / sharpness)
    spec_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=cfg.hop_length
    )[0]
    sc_mean, sc_std, sc_max = _safe_stats(spec_centroid)

    # Simple speech-rate proxy:
    # Use librosa.beat.tempo as a crude surrogate (pulsing rate of the signal)
    try:
        tempo, _ = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=cfg.hop_length
        )
        speech_rate_proxy = float(tempo)
    except Exception:
        speech_rate_proxy = 0.0

    features = {
        # Pitch
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_max": f0_max,
        # Energy
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_max": rms_max,
        # Zero-crossing
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "zcr_max": zcr_max,
        # Spectral centroid
        "sc_mean": sc_mean,
        "sc_std": sc_std,
        "sc_max": sc_max,
        # Speech rate proxy
        "speech_rate_proxy": speech_rate_proxy,
    }

    return features


def prosodic_feature_vector(
    prosodic_dict: Dict[str, float],
    feature_order: Tuple[str, ...] | None = None,
) -> np.ndarray:
    """
    Convert a prosodic feature dict into a fixed-order vector.

    Args:
        prosodic_dict: Output from `extract_prosodic_features`.
        feature_order: Optional explicit ordering of keys. If None,
                       uses a default stable order.

    Returns:
        1D numpy array of shape (num_features,).
    """
    if feature_order is None:
        feature_order = (
            "f0_mean",
            "f0_std",
            "f0_max",
            "rms_mean",
            "rms_std",
            "rms_max",
            "zcr_mean",
            "zcr_std",
            "zcr_max",
            "sc_mean",
            "sc_std",
            "sc_max",
            "speech_rate_proxy",
        )

    vec = np.array(
        [float(prosodic_dict.get(k, 0.0)) for k in feature_order],
        dtype=np.float32,
    )
    return vec


