"""
DistilHuBERT-based Audio Emotion Model
--------------------------------------

This module defines:
- `DistilHuBERTEmotionModel`: a PyTorch module that combines
  DistilHuBERT embeddings with handcrafted prosodic features.
- `AudioEmotionInference`: a lightweight wrapper for inference that returns
  both logits and an embedding suitable for multi-modal fusion with text.

The model is designed to be trained on CREMA-D labels but general enough
to be reused on other datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor


@dataclass
class AudioEmotionConfig:
    """
    Configuration for the DistilHuBERT emotion model.

    Attributes:
        base_model_name: Hugging Face model id for DistilHuBERT.
        num_labels: Number of emotion classes.
        prosodic_dim: Dimension of prosodic feature vector.
        hidden_dim: Hidden size for classifier MLP.
        dropout: Dropout probability.
    """

    base_model_name: str = "facebook/wav2vec2-base"  # Using Wav2Vec2 which is more stable and well-documented
    num_labels: int = 6
    prosodic_dim: int = 13  # from prosodic_feature_vector default
    hidden_dim: int = 256
    dropout: float = 0.2


class DistilHuBERTEmotionModel(nn.Module):
    """
    DistilHuBERT + Prosodic Features for Emotion Classification.

    Inputs:
        - waveforms: Tensor of shape (batch, time) in float32 (16 kHz).
        - prosodic_feats: Tensor of shape (batch, prosodic_dim).
    Outputs:
        - logits: Tensor of shape (batch, num_labels).
        - pooled_embedding: Tensor of shape (batch, hidden_dim),
          which can be used as an audio embedding for multi-modal fusion.
    """

    def __init__(self, cfg: AudioEmotionConfig):
        super().__init__()
        self.cfg = cfg

        # Wav2Vec2 backbone (more stable than HuBERT for this use case)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            cfg.base_model_name
        )
        self.backbone = AutoModel.from_pretrained(cfg.base_model_name)

        hubert_dim = self.backbone.config.hidden_size

        # Combine backbone pooled output with prosodic features
        combined_dim = hubert_dim + cfg.prosodic_dim

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_labels)

    def forward(
        self, waveforms: torch.Tensor, prosodic_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            waveforms: Tensor (batch, time) float32.
            prosodic_feats: Tensor (batch, prosodic_dim) float32.

        Returns:
            logits, pooled_embedding
        """
        # Prepare inputs for Wav2Vec2 feature extractor
        # Convert waveforms to numpy arrays (Wav2Vec2 expects raw audio as numpy)
        waveforms_np = waveforms.cpu().numpy() if isinstance(waveforms, torch.Tensor) else waveforms
        # Handle batch dimension - convert to list of 1D arrays
        if waveforms_np.ndim == 2:
            waveforms_list = [waveforms_np[i].astype(np.float32) for i in range(waveforms_np.shape[0])]
        elif waveforms_np.ndim == 1:
            waveforms_list = [waveforms_np.astype(np.float32)]
        else:
            waveforms_list = [waveforms_np.flatten().astype(np.float32)]
        
        inputs = self.feature_extractor(
            waveforms_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs["input_values"].to(waveforms.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(waveforms.device)

        outputs = self.backbone(
            input_values, attention_mask=attention_mask
        )
        # Mean pool over time dimension
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            masked = last_hidden * mask
            pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = last_hidden.mean(dim=1)

        # Concatenate prosodic features
        combined = torch.cat([pooled, prosodic_feats], dim=-1)

        hidden = self.mlp(combined)
        logits = self.classifier(hidden)

        return logits, hidden


class AudioEmotionInference:
    """
    Convenience wrapper for running inference with the emotion model.

    It exposes a simple `predict` API that returns:
        - probabilities over emotions
        - predicted label indices
        - pooled embeddings for fusion with text models
    """

    def __init__(
        self,
        model: DistilHuBERTEmotionModel,
        id2label: Dict[int, str],
        device: Optional[str] = None,
    ):
        self.model = model
        self.id2label = id2label
        if device is None:
            # Get best available device (MPS > CUDA > CPU)
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
    def predict(
        self,
        waveforms: torch.Tensor,
        prosodic_feats: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass and return structured outputs.

        Args:
            waveforms: (batch, time) float32 tensor.
            prosodic_feats: (batch, prosodic_dim) float32 tensor.

        Returns:
            {
                "logits": (B, C),
                "probs": (B, C),
                "pred_ids": (B,),
                "pred_labels": List[str],
                "embeddings": (B, hidden_dim),
            }
        """
        waveforms = waveforms.to(self.device)
        prosodic_feats = prosodic_feats.to(self.device)

        logits, embeddings = self.model(waveforms, prosodic_feats)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        pred_labels = [self.id2label[int(i)] for i in pred_ids]

        return {
            "logits": logits,
            "probs": probs,
            "pred_ids": pred_ids,
            "pred_labels": pred_labels,
            "embeddings": embeddings,
        }


