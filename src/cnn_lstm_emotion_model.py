"""
CNN + LSTM Emotion Model with Enhanced Librosa Features
-------------------------------------------------------

This is a classic but highly effective approach for emotion recognition:
- Extracts comprehensive audio features (MFCCs, chroma, spectral, prosodic)
- Uses CNN to learn local patterns in feature sequences
- Uses LSTM to capture temporal dynamics
- Often outperforms transformer-based models for emotion recognition

Based on proven architectures from emotion recognition literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import librosa


@dataclass
class CNNLSTMConfig:
    """
    Configuration for CNN+LSTM emotion model.
    
    Attributes:
        num_labels: Number of emotion classes
        feature_dim: Dimension of input features (MFCCs + other features)
        cnn_channels: Number of CNN channels
        lstm_hidden: LSTM hidden dimension
        lstm_layers: Number of LSTM layers
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """
    num_labels: int = 6
    feature_dim: int = 39  # 13 MFCCs + 12 chroma + 7 spectral + 7 prosodic
    cnn_channels: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout: float = 0.3
    use_batch_norm: bool = True


def extract_comprehensive_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract comprehensive audio features for emotion recognition.
    
    Features extracted:
    - MFCCs (13): Mel-frequency cepstral coefficients
    - Chroma (12): Chroma features (pitch class)
    - Spectral (7): Spectral contrast, rolloff, bandwidth
    - Prosodic (7): Pitch, energy, ZCR statistics
    
    Returns:
        Feature matrix of shape (time_frames, feature_dim)
    """
    # Ensure float32 and normalize
    y = librosa.util.normalize(audio.astype(np.float32))
    
    # Frame parameters
    n_fft = 2048
    hop_length = 512
    n_mfcc = 13
    n_chroma = 12
    
    features_list = []
    
    # 1. MFCCs (13 features) - most important for emotion
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    features_list.append(mfccs)
    
    # 2. Chroma features (12 features) - pitch class information
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    # Ensure chroma has same time dimension as MFCCs
    target_time_frames = mfccs.shape[1]
    if chroma.shape[1] < target_time_frames:
        pad_width = ((0, 0), (0, target_time_frames - chroma.shape[1]))
        chroma = np.pad(chroma, pad_width, mode='constant')
    elif chroma.shape[1] > target_time_frames:
        chroma = chroma[:, :target_time_frames]
    features_list.append(chroma)
    
    # 3. Spectral features - ensure all have same time dimension
    target_time_frames = mfccs.shape[1]
    
    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    # Pad or trim to match target
    if spectral_contrast.shape[1] < target_time_frames:
        pad_width = ((0, 0), (0, target_time_frames - spectral_contrast.shape[1]))
        spectral_contrast = np.pad(spectral_contrast, pad_width, mode='constant')
    elif spectral_contrast.shape[1] > target_time_frames:
        spectral_contrast = spectral_contrast[:, :target_time_frames]
    features_list.append(spectral_contrast)
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    if rolloff.shape[1] < target_time_frames:
        pad_width = ((0, 0), (0, target_time_frames - rolloff.shape[1]))
        rolloff = np.pad(rolloff, pad_width, mode='constant')
    elif rolloff.shape[1] > target_time_frames:
        rolloff = rolloff[:, :target_time_frames]
    features_list.append(rolloff)
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    if bandwidth.shape[1] < target_time_frames:
        pad_width = ((0, 0), (0, target_time_frames - bandwidth.shape[1]))
        bandwidth = np.pad(bandwidth, pad_width, mode='constant')
    elif bandwidth.shape[1] > target_time_frames:
        bandwidth = bandwidth[:, :target_time_frames]
    features_list.append(bandwidth)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop_length
    )
    if zcr.shape[1] < target_time_frames:
        pad_width = ((0, 0), (0, target_time_frames - zcr.shape[1]))
        zcr = np.pad(zcr, pad_width, mode='constant')
    elif zcr.shape[1] > target_time_frames:
        zcr = zcr[:, :target_time_frames]
    features_list.append(zcr)
    
    # 4. Prosodic features (7 features) - frame-level
    # Pitch (F0) using YIN
    try:
        f0 = librosa.yin(
            y, fmin=50, fmax=500, sr=sr,
            frame_length=n_fft, hop_length=hop_length
        )
        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)
    except:
        f0 = np.zeros((mfccs.shape[1],))
    
    # Energy (RMS)
    rms = librosa.feature.rms(
        y=y, frame_length=n_fft, hop_length=hop_length
    )[0]
    
    # Normalize f0 and rms to match other features
    if f0.max() > 0:
        f0 = f0 / f0.max()
    if rms.max() > 0:
        rms = rms / rms.max()
    
    # Stack prosodic features
    prosodic = np.stack([f0, rms], axis=0)
    
    # Pad or trim to match MFCC length
    target_length = mfccs.shape[1]
    if prosodic.shape[1] < target_length:
        prosodic = np.pad(prosodic, ((0, 0), (0, target_length - prosodic.shape[1])), mode='constant')
    elif prosodic.shape[1] > target_length:
        prosodic = prosodic[:, :target_length]
    
    features_list.append(prosodic)
    
    # Concatenate all features
    features = np.concatenate(features_list, axis=0)  # (feature_dim, time_frames)
    features = features.T  # (time_frames, feature_dim) - time first for LSTM
    
    # Normalize features
    features = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-8)
    
    return features.astype(np.float32)


class CNNLSTMEmotionModel(nn.Module):
    """
    CNN + LSTM model for emotion recognition.
    
    Architecture:
    1. CNN layers: Learn local patterns in feature sequences
    2. LSTM layers: Capture temporal dynamics
    3. Attention/Global pooling: Aggregate temporal information
    4. Classifier: Final emotion prediction
    """
    
    def __init__(self, cfg: CNNLSTMConfig):
        super().__init__()
        self.cfg = cfg
        
        # CNN layers for local pattern learning
        self.cnn = nn.Sequential(
            # Conv1D: (batch, features, time) -> (batch, channels, time)
            nn.Conv1d(
                in_channels=cfg.feature_dim,
                out_channels=cfg.cnn_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(cfg.cnn_channels) if cfg.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            
            nn.Conv1d(
                in_channels=cfg.cnn_channels,
                out_channels=cfg.cnn_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(cfg.cnn_channels) if cfg.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            
            # Max pooling
            nn.MaxPool1d(kernel_size=2),
        )
        
        # LSTM for temporal modeling
        # Input: (batch, time, cnn_channels) after transpose
        self.lstm = nn.LSTM(
            input_size=cfg.cnn_channels,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0,
            bidirectional=True,  # Bidirectional for better context
        )
        
        # Attention mechanism for temporal aggregation
        self.attention = nn.Sequential(
            nn.Linear(cfg.lstm_hidden * 2, cfg.lstm_hidden),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(cfg.lstm_hidden, 1),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg.lstm_hidden * 2, cfg.lstm_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.lstm_hidden, cfg.num_labels),
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: (batch, time_frames, feature_dim) tensor
        
        Returns:
            logits: (batch, num_labels)
            embedding: (batch, lstm_hidden * 2) for fusion
        """
        batch_size = features.shape[0]
        
        # CNN expects (batch, channels, time)
        # Features are (batch, time, channels), so transpose
        x = features.transpose(1, 2)  # (batch, feature_dim, time)
        
        # CNN layers
        x = self.cnn(x)  # (batch, cnn_channels, time')
        
        # Transpose back for LSTM: (batch, time', cnn_channels)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, time', lstm_hidden * 2)
        
        # Attention-based pooling
        attention_weights = self.attention(lstm_out)  # (batch, time', 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        pooled = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, lstm_hidden * 2)
        
        # Alternative: Use last hidden state if attention fails
        if pooled.isnan().any():
            # Use mean pooling as fallback
            pooled = lstm_out.mean(dim=1)
        
        # Get embedding for fusion
        embedding = pooled
        
        # Classifier
        logits = self.classifier(pooled)
        
        return logits, embedding


class CNNLSTMInference:
    """Inference wrapper for CNN+LSTM model"""
    
    def __init__(
        self,
        model: CNNLSTMEmotionModel,
        id2label: Dict[int, str],
        device: Optional[str] = None,
    ):
        self.model = model
        self.id2label = id2label
        
        if device is None:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, waveforms: torch.Tensor, prosodic_feats: torch.Tensor = None, sr: int = 16000) -> Dict[str, torch.Tensor]:
        """
        Predict emotion from audio.
        
        Compatible with AudioEmotionInference interface for fusion model.
        
        Args:
            waveforms: Audio waveforms tensor (batch, time) or numpy array
            prosodic_feats: Ignored (for compatibility with Wav2Vec2 interface)
            sr: Sample rate (default 16000)
        
        Returns:
            Dictionary with predictions and probabilities
        """
        # Handle both torch.Tensor and numpy array inputs
        if isinstance(waveforms, torch.Tensor):
            # Convert to numpy for feature extraction
            if waveforms.is_cuda or (hasattr(torch.backends, 'mps') and waveforms.device.type == 'mps'):
                audio_np = waveforms.cpu().numpy()
            else:
                audio_np = waveforms.numpy()
            
            # Handle batch dimension
            if len(audio_np.shape) > 1:
                # Take first sample if batch (fusion model may pass batch)
                audio_np = audio_np[0] if audio_np.shape[0] == 1 else audio_np[0]
        else:
            audio_np = waveforms
        
        # Ensure 1D
        if len(audio_np.shape) > 1:
            audio_np = audio_np.flatten()
        
        # Extract features
        features = extract_comprehensive_features(audio_np, sr)
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        features_tensor = features_tensor.to(self.device)
        
        # Predict
        logits, embedding = self.model(features_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1)[0]
        pred_label = self.id2label[int(pred_id)]
        
        return {
            "logits": logits,
            "probs": probs,
            "pred_ids": pred_id.unsqueeze(0) if pred_id.dim() == 0 else pred_id,  # Ensure batch dimension
            "pred_labels": [pred_label],  # List for compatibility with AudioEmotionInference
            "pred_id": pred_id,  # Keep for backward compatibility
            "pred_label": pred_label,  # Keep for backward compatibility
            "embeddings": embedding,  # Use "embeddings" (plural) for compatibility
        }


__all__ = [
    "CNNLSTMConfig",
    "CNNLSTMEmotionModel",
    "CNNLSTMInference",
    "extract_comprehensive_features",
]

