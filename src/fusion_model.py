"""
Multimodal Fusion Model for Audio-Text Emotion Detection
---------------------------------------------------------

This module implements a fusion layer that combines:
- Audio embeddings from DistilHuBERTEmotionModel (Wav2Vec2-base + prosodic features)
- Text embeddings from RoBERTa-base with LoRA

The fusion model uses:
1. Early fusion: Concatenate audio and text embeddings
2. Late fusion: Combine predictions from both modalities
3. Attention-based fusion: Learnable attention weights for each modality

Best models selected:
- Audio: DistilHuBERTEmotionModel (Wav2Vec2-base) from audio_emotion_model_improved
- Text: RoBERTa-base with LoRA for text emotion classification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class FusionConfig:
    """
    Configuration for the multimodal fusion model.
    
    Attributes:
        audio_embedding_dim: Dimension of audio embeddings (from DistilHuBERTEmotionModel)
        text_embedding_dim: Dimension of text embeddings (from RoBERTa, typically 768)
        num_labels: Number of emotion classes (6 for CREMA-D/GoEmotions)
        fusion_hidden_dim: Hidden dimension for fusion layers
        dropout: Dropout probability
        fusion_type: Type of fusion ('early', 'late', 'attention', 'weighted')
    
    Note: Default values are provided for convenience, but in production,
    these should be set from config.FUSION_CONFIG in config.py.
    """
    audio_embedding_dim: int = 256  # From AudioEmotionConfig.hidden_dim
    text_embedding_dim: int = 768  # RoBERTa-base hidden size
    num_labels: int = 6
    fusion_hidden_dim: int = 512
    dropout: float = 0.3
    fusion_type: str = "attention"  # Options: 'early', 'late', 'attention', 'weighted'
    # Note: Override via config.FUSION_CONFIG["fusion_type"] in production


class AttentionFusion(nn.Module):
    """
    Attention-based fusion layer that learns to weight audio and text embeddings.
    """
    
    def __init__(self, audio_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        
        # Project both embeddings to same dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer norm and feedforward
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
    
    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_emb: (batch, audio_dim)
            text_emb: (batch, text_dim)
        
        Returns:
            fused_emb: (batch, hidden_dim)
        """
        # Project to same dimension
        audio_proj = self.audio_proj(audio_emb)  # (B, hidden_dim)
        text_proj = self.text_proj(text_emb)  # (B, hidden_dim)
        
        # Stack for attention: (B, 2, hidden_dim)
        stacked = torch.stack([audio_proj, text_proj], dim=1)
        
        # Self-attention
        attn_out, _ = self.attention(stacked, stacked, stacked)  # (B, 2, hidden_dim)
        
        # Mean pool over modalities
        fused = attn_out.mean(dim=1)  # (B, hidden_dim)
        
        # Residual + norm + feedforward
        fused = self.norm(fused + audio_proj)  # Residual connection
        fused = fused + self.ff(fused)  # Feedforward with residual
        
        return fused


class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion model that combines audio and text emotion predictions.
    
    Supports multiple fusion strategies:
    - early: Concatenate embeddings and classify
    - late: Average predictions from both modalities
    - attention: Use attention mechanism to learn optimal combination
    - weighted: Learnable weighted combination of predictions
    """
    
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        
        if cfg.fusion_type == "early":
            # Simple concatenation + MLP
            combined_dim = cfg.audio_embedding_dim + cfg.text_embedding_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(combined_dim, cfg.fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            )
            self.classifier = nn.Linear(cfg.fusion_hidden_dim, cfg.num_labels)
            
        elif cfg.fusion_type == "late":
            # Late fusion: combine predictions from trained models directly
            # No need for separate classifiers - we use the actual model logits
            # Learnable combination weights (can be overridden by config)
            # Default: favor audio slightly more (0.6/0.4) since audio is typically more reliable
            self.combination_weights = nn.Parameter(torch.tensor([0.2, 0.8]))
            
        elif cfg.fusion_type == "attention":
            # Attention-based fusion
            self.fusion_layer = AttentionFusion(
                cfg.audio_embedding_dim,
                cfg.text_embedding_dim,
                cfg.fusion_hidden_dim
            )
            self.classifier = nn.Sequential(
                nn.Linear(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.fusion_hidden_dim, cfg.num_labels),
            )
            
        elif cfg.fusion_type == "weighted":
            # Learnable weighted combination with separate projections
            self.audio_proj = nn.Linear(cfg.audio_embedding_dim, cfg.fusion_hidden_dim)
            self.text_proj = nn.Linear(cfg.text_embedding_dim, cfg.fusion_hidden_dim)
            self.audio_weight = nn.Parameter(torch.tensor(0.3))
            self.text_weight = nn.Parameter(torch.tensor(0.7))
            self.classifier = nn.Sequential(
                nn.Linear(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.fusion_hidden_dim, cfg.num_labels),
            )
        else:
            raise ValueError(f"Unknown fusion type: {cfg.fusion_type}")
    
    def forward(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        audio_logits: Optional[torch.Tensor] = None,
        text_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through fusion model.
        
        Args:
            audio_emb: Audio embeddings (batch, audio_embedding_dim)
            text_emb: Text embeddings (batch, text_embedding_dim)
            audio_logits: Optional audio predictions (batch, num_labels)
            text_logits: Optional text predictions (batch, num_labels)
        
        Returns:
            fused_logits: Combined predictions (batch, num_labels)
            aux_outputs: Dictionary with intermediate outputs
        """
        aux_outputs = {}
        
        if self.cfg.fusion_type == "early":
            # Concatenate embeddings
            combined = torch.cat([audio_emb, text_emb], dim=-1)
            fused_emb = self.fusion_layer(combined)
            fused_logits = self.classifier(fused_emb)
            
        elif self.cfg.fusion_type == "late":
            # Late fusion: combine predictions from trained models
            # audio_logits and text_logits MUST be provided (from actual models)
            if audio_logits is None or text_logits is None:
                raise ValueError("Late fusion requires both audio_logits and text_logits from trained models")
            
            # Ensure both have the same number of labels
            if audio_logits.shape[-1] != text_logits.shape[-1]:
                raise ValueError(f"Label count mismatch: audio={audio_logits.shape[-1]}, text={text_logits.shape[-1]}")
            
            # Normalize weights (learnable combination)
            weights = torch.softmax(self.combination_weights, dim=0)
            
            # Combine logits with learned weights
            # Note: If weights are untrained (default 0.5/0.5), this is simple averaging
            fused_logits = weights[0] * audio_logits + weights[1] * text_logits
            
            aux_outputs["audio_logits"] = audio_logits
            aux_outputs["text_logits"] = text_logits
            aux_outputs["weights"] = weights
            aux_outputs["audio_weight"] = weights[0]
            aux_outputs["text_weight"] = weights[1]
            
            
        elif self.cfg.fusion_type == "attention":
            # Attention-based fusion
            fused_emb = self.fusion_layer(audio_emb, text_emb)
            fused_logits = self.classifier(fused_emb)
            aux_outputs["fused_embedding"] = fused_emb
            
        elif self.cfg.fusion_type == "weighted":
            # Weighted combination
            audio_proj = self.audio_proj(audio_emb)
            text_proj = self.text_proj(text_emb)
            
            # Normalize weights
            audio_w = torch.sigmoid(self.audio_weight)
            text_w = torch.sigmoid(self.text_weight)
            total_w = audio_w + text_w
            audio_w = audio_w / total_w
            text_w = text_w / total_w
            
            fused_emb = audio_w * audio_proj + text_w * text_proj
            fused_logits = self.classifier(fused_emb)
            
            aux_outputs["audio_weight"] = audio_w
            aux_outputs["text_weight"] = text_w
            aux_outputs["fused_embedding"] = fused_emb
        
        return fused_logits, aux_outputs


class MultimodalEmotionInference:
    """
    Convenience wrapper for multimodal emotion inference.
    
    Combines:
    - AudioEmotionInference (audio model)
    - TextEmotionPredictor (text model)
    - MultimodalFusionModel (fusion layer)
    """
    
    def __init__(
        self,
        audio_model,
        text_model: Optional[nn.Module],
        fusion_model: MultimodalFusionModel,
        id2label: Dict[int, str],
        device: Optional[str] = None,
        audio_id2label: Optional[Dict[int, str]] = None,
        text_id2label: Optional[Dict[int, str]] = None,
    ):
        self.audio_model = audio_model
        self.text_model = text_model
        self.fusion_model = fusion_model
        self.id2label = id2label  # Canonical order
        self.audio_id2label = audio_id2label or audio_model.id2label
        self.text_id2label = text_id2label
        
        if device is None:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        
        self.fusion_model.to(self.device)
        self.fusion_model.eval()
    
    @torch.no_grad()
    def predict(
        self,
        waveforms: torch.Tensor,
        prosodic_feats: torch.Tensor,
        text: Optional[str] = None,
        text_tokenizer=None,
        return_individual: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run multimodal emotion prediction.
        
        Args:
            waveforms: Audio waveforms (batch, time)
            prosodic_feats: Prosodic features (batch, prosodic_dim)
            text: Optional text transcription
            text_tokenizer: Tokenizer for text model
            return_individual: Whether to return individual modality predictions
        
        Returns:
            Dictionary with predictions and probabilities
        """
        waveforms = waveforms.to(self.device)
        # Only move prosodic_feats to device if it's not None
        if prosodic_feats is not None:
            prosodic_feats = prosodic_feats.to(self.device)
        
        # Audio prediction
        # CNN+LSTM ignores prosodic_feats, Wav2Vec2 uses them
        audio_results = self.audio_model.predict(waveforms, prosodic_feats)
        audio_emb = audio_results["embeddings"]
        audio_logits = audio_results["logits"]
        audio_probs = audio_results["probs"]
        
        # Text prediction (if available)
        text_emb = None
        text_logits = None
        text_probs = None
        
        if self.text_model is not None and text is not None and text_tokenizer is not None:
            # Tokenize text
            if isinstance(text, str):
                text_list = [text]
            else:
                text_list = text
            
            enc = text_tokenizer(
                text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)
            
            # Get text embeddings and predictions
            with torch.no_grad():
                # Get predictions from full model first
                text_outputs = self.text_model(**enc, output_hidden_states=True)
                text_logits = text_outputs.logits
                
                # Extract embeddings from hidden states
                # Handle different model structures (PEFT, standard, etc.)
                if hasattr(text_outputs, 'hidden_states') and text_outputs.hidden_states is not None:
                    # Use CLS token (first token) from last hidden state
                    text_emb = text_outputs.hidden_states[-1][:, 0, :]  # (B, 768)
                else:
                    # Fallback: try to access base model directly
                    if hasattr(self.text_model, 'roberta'):
                        base_model = self.text_model.roberta
                    elif hasattr(self.text_model, 'base_model'):
                        # PEFT model structure
                        if hasattr(self.text_model.base_model, 'model'):
                            base_model = self.text_model.base_model.model
                        elif hasattr(self.text_model.base_model, 'roberta'):
                            base_model = self.text_model.base_model.roberta
                        else:
                            base_model = self.text_model.base_model
                    else:
                        base_model = self.text_model.model if hasattr(self.text_model, 'model') else self.text_model
                    
                    # Get embeddings from base model
                    base_outputs = base_model(**enc, output_hidden_states=True)
                    text_emb = base_outputs.last_hidden_state[:, 0, :]  # (B, 768)
            
            text_probs = torch.softmax(text_logits, dim=-1)
        else:
            # Use zero embeddings if text not available (fusion will rely on audio)
            text_emb = torch.zeros(
                audio_emb.shape[0],
                self.fusion_model.cfg.text_embedding_dim,
                device=self.device
            )
        
        # For late fusion, we need to remap logits to canonical order before fusion
        # For other fusion types, we use embeddings which don't depend on label order
        audio_logits_for_fusion = audio_logits
        text_logits_for_fusion = text_logits
        
        if self.fusion_model.cfg.fusion_type == "late":
            # Remap logits to canonical order for late fusion
            canonical_order = list(self.id2label.values())
            num_canonical_labels = len(canonical_order)
            
            # Remap audio logits to canonical order
            audio_remapped = torch.zeros(
                audio_logits.shape[0], 
                num_canonical_labels, 
                device=audio_logits.device, 
                dtype=audio_logits.dtype
            )
            for audio_idx, audio_label in self.audio_id2label.items():
                if audio_label in canonical_order and audio_idx < audio_logits.shape[1]:
                    canonical_idx = canonical_order.index(audio_label)
                    audio_remapped[:, canonical_idx] = audio_logits[:, audio_idx]
            audio_logits_for_fusion = audio_remapped
            
            # Remap text logits if available
            if text_logits is not None and self.text_id2label:
                text_remapped = torch.zeros(
                    text_logits.shape[0],
                    num_canonical_labels,
                    device=text_logits.device,
                    dtype=text_logits.dtype
                )
                for text_idx, text_label in self.text_id2label.items():
                    if text_label in canonical_order and text_idx < text_logits.shape[1]:
                        canonical_idx = canonical_order.index(text_label)
                        text_remapped[:, canonical_idx] = text_logits[:, text_idx]
                text_logits_for_fusion = text_remapped
            else:
                # If no text, create zero logits in canonical order
                text_logits_for_fusion = torch.zeros(
                    audio_logits.shape[0],
                    num_canonical_labels,
                    device=audio_logits.device,
                    dtype=audio_logits.dtype
                )
        
        # Fusion prediction
        fused_logits, aux_outputs = self.fusion_model(
            audio_emb,
            text_emb,
            audio_logits=audio_logits_for_fusion if (return_individual or self.fusion_model.cfg.fusion_type == "late") else None,
            text_logits=text_logits_for_fusion if (return_individual or self.fusion_model.cfg.fusion_type == "late") else None,
        )
        fused_probs = torch.softmax(fused_logits, dim=-1)
        fused_pred_ids = torch.argmax(fused_probs, dim=-1)
        fused_pred_labels = [self.id2label[int(i)] for i in fused_pred_ids]
        
        result = {
            "fused_logits": fused_logits,
            "fused_probs": fused_probs,
            "fused_pred_ids": fused_pred_ids,
            "fused_pred_labels": fused_pred_labels,
        }
        
        if return_individual:
            result["audio_logits"] = audio_logits
            result["audio_probs"] = audio_probs
            result["audio_pred_labels"] = audio_results["pred_labels"]
            
            if text_logits is not None:
                result["text_logits"] = text_logits
                result["text_probs"] = text_probs
                text_pred_ids = torch.argmax(text_probs, dim=-1)
                result["text_pred_labels"] = [self.id2label[int(i)] for i in text_pred_ids]
        
        result.update(aux_outputs)
        return result


__all__ = [
    "FusionConfig",
    "MultimodalFusionModel",
    "MultimodalEmotionInference",
    "AttentionFusion",
]

