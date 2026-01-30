"""
Dataset and stratified splitting utilities for text emotion classification.

This module is responsible ONLY for:
- Loading CSV files with columns: `text`, `label`
- Encoding string labels to integer IDs
- Creating stratified train/validation splits (important for imbalanced GoEmotions)
- Providing PyTorch Datasets for training and validation

Model definition and training loops live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

import config


@dataclass
class LabelEncoding:
    label2id: Dict[str, int]
    id2label: Dict[int, str]


class TextEmotionDataset(Dataset):
    """
    Simple Dataset wrapping tokenized texts and integer labels.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: RobertaTokenizerFast,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def build_label_encoding() -> LabelEncoding:
    """
    Create a consistent label<->id mapping based on config.CANONICAL_EMOTIONS.
    """
    label2id = {lbl: i for i, lbl in enumerate(config.CANONICAL_EMOTIONS)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return LabelEncoding(label2id=label2id, id2label=id2label)


def stratified_train_val_split(
    csv_path: Path,
    val_ratio: float = 0.1,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split of a `text,label` CSV into train/val DataFrames.
    """
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=random_state
    )
    (train_idx, val_idx) = next(splitter.split(df["text"], df["label"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def create_datasets_from_csv(
    csv_path: Path,
    tokenizer: RobertaTokenizerFast,
    max_length: int = 128,
    val_ratio: float = 0.1,
) -> Tuple[TextEmotionDataset, TextEmotionDataset, LabelEncoding]:
    """
    Convenience function:
    - Reads a CSV
    - Performs a stratified train/val split
    - Encodes labels to integer IDs
    - Returns train and val Datasets + label encoding.
    """
    label_enc = build_label_encoding()
    train_df, val_df = stratified_train_val_split(csv_path, val_ratio=val_ratio)

    def encode_labels(series: pd.Series) -> List[int]:
        return [label_enc.label2id[str(lbl)] for lbl in series]

    train_labels = encode_labels(train_df["label"])
    val_labels = encode_labels(val_df["label"])

    train_ds = TextEmotionDataset(
        texts=train_df["text"].tolist(),
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_ds = TextEmotionDataset(
        texts=val_df["text"].tolist(),
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return train_ds, val_ds, label_enc


__all__ = [
    "TextEmotionDataset",
    "LabelEncoding",
    "build_label_encoding",
    "stratified_train_val_split",
    "create_datasets_from_csv",
]




