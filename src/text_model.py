"""
RoBERTa-based text emotion classifier with optional LoRA.

This module is responsible for:
- Loading a RoBERTa model and tokenizer
- Attaching a 6-way classification head
- Optionally wrapping attention layers with LoRA (Mode A = default: base frozen)
- Providing a simple forward API for training/evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

import config

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


NUM_LABELS = len(config.CANONICAL_EMOTIONS)


@dataclass
class TextModelArtifacts:
    model: RobertaForSequenceClassification
    tokenizer: RobertaTokenizerFast
    id2label: Dict[int, str]
    label2id: Dict[str, int]


def load_roberta_text_model(
    model_name: str = "roberta-base",
    use_lora: bool = True,
    freeze_base: bool = True,
) -> TextModelArtifacts:
    """
    Load RoBERTa + classification head for 6-way emotion classification.

    Mode A (recommended default):
        use_lora=True, freeze_base=True
        → Freeze base RoBERTa weights and train only LoRA+head.

    Mode B (optional later):
        use_lora=True, freeze_base=False
        → Allow some base layers to update if you see a clear ceiling.
    """
    label2id = {lbl: i for i, lbl in enumerate(config.CANONICAL_EMOTIONS)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    if use_lora and get_peft_model is not None and LoraConfig is not None:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)

    if freeze_base:
        for name, param in model.named_parameters():
            # Keep LoRA and classifier head trainable, freeze others
            if "classifier" in name or "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return TextModelArtifacts(
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        label2id=label2id,
    )


class TextEmotionPredictor(nn.Module):
    """
    Thin wrapper around RobertaForSequenceClassification for inference convenience.
    """

    def __init__(self, artifacts: TextModelArtifacts) -> None:
        super().__init__()
        self.model = artifacts.model
        self.tokenizer = artifacts.tokenizer
        self.id2label = artifacts.id2label

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.inference_mode()
    def predict_text(self, text: str, device: torch.device) -> Dict[str, float]:
        """
        Convenience helper: single-text prediction returning label + probabilities.
        """
        self.eval()
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        outputs = self.model(**enc)
        probs = outputs.logits.softmax(dim=-1).squeeze(0)
        pred_id = int(probs.argmax().item())
        pred_label = self.id2label[pred_id]

        return {
            "label": pred_label,
            "probabilities": {self.id2label[i]: float(p) for i, p in enumerate(probs)},
        }


__all__ = [
    "TextModelArtifacts",
    "load_roberta_text_model",
    "TextEmotionPredictor",
]




