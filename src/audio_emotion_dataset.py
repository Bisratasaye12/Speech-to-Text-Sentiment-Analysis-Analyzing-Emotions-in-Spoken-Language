"""
PyTorch Dataset for CREMA-D Audio Emotion Fine-tuning
-----------------------------------------------------

This dataset:
- Reads CREMA-D split CSVs (train/val/test) created earlier in the pipeline.
- Loads processed audio waveforms.
- Extracts prosodic features using `audio_features`.
- Returns tensors ready for DistilHuBERT + prosodic fusion.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
from pathlib import Path

import config
from .audio_features import extract_prosodic_features, prosodic_feature_vector


class CREMAAudioEmotionDataset(Dataset):
    """
    Dataset for supervised audio emotion classification on CREMA-D.

    Expected CSV columns (per existing splits/metadata):
        - 'filename'
        - 'emotion'   (e.g., 'anger', 'disgust', etc.)
        - 'processed_filepath' or 'filepath'
    """

    def __init__(
        self,
        csv_path: Path,
        label2id: Dict[str, int],
        audio_col_candidates: Tuple[str, ...] = (
            "processed_filepath",
            "filepath",
        ),
        sr: Optional[int] = None,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}

        # Determine audio path column
        for col in audio_col_candidates:
            if col in self.df.columns:
                self.audio_col = col
                break
        else:
            raise ValueError(
                f"None of audio columns {audio_col_candidates} found in {csv_path}"
            )

        if "emotion" not in self.df.columns:
            raise ValueError(
                f"'emotion' column not found in {csv_path}. "
                "Ensure CREMA-D metadata is present."
            )

        self.sr = sr or config.AUDIO_CONFIG["target_sample_rate"]

        # Pre-resolve absolute paths
        self.paths: List[Path] = []
        for p in self.df[self.audio_col].tolist():
            path = Path(p)
            if not path.is_absolute():
                path = config.PROCESSED_DATA_DIR / path
            self.paths.append(path)

        # Pre-compute label ids
        self.labels: List[int] = []
        for emo in self.df["emotion"].tolist():
            if emo not in self.label2id:
                raise ValueError(
                    f"Unknown emotion '{emo}' in {csv_path}. "
                    f"Expected one of: {list(self.label2id.keys())}"
                )
            self.labels.append(self.label2id[emo])

    def __len__(self) -> int:
        return len(self.paths)

    def _load_audio(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio, _ = librosa.load(
            path,
            sr=self.sr,
            mono=config.AUDIO_CONFIG["mono"],
        )
        return audio.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        label = self.labels[idx]

        audio = self._load_audio(path)
        prosodic_dict = extract_prosodic_features(audio, sr=self.sr)
        prosodic_vec = prosodic_feature_vector(prosodic_dict)

        # Convert to tensors (ensure float32 for MPS compatibility)
        waveform_tensor = torch.from_numpy(audio).float()  # (time,) - float32
        prosodic_tensor = torch.from_numpy(prosodic_vec).float()  # (feat_dim,) - float32
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "waveform": waveform_tensor,
            "prosodic": prosodic_tensor,
            "label": label_tensor,
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length waveforms.

    - Pads waveforms to the maximum length in the batch.
    - Stacks prosodic features and labels.
    """
    waveforms = [b["waveform"] for b in batch]
    prosodics = [b["prosodic"] for b in batch]
    labels = [b["label"] for b in batch]

    # Pad waveforms (ensure float32 for MPS compatibility)
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)
    padded = []
    for w in waveforms:
        w = w.float()  # Ensure float32
        if w.shape[0] < max_len:
            pad = torch.zeros(max_len - w.shape[0], dtype=torch.float32)
            w = torch.cat([w, pad], dim=0)
        padded.append(w)
    waveforms_tensor = torch.stack(padded, dim=0)  # (B, T)

    # Ensure prosodics are float32 for MPS compatibility
    prosodics = [p.float() for p in prosodics]
    prosodics_tensor = torch.stack(prosodics, dim=0)  # (B, F)
    labels_tensor = torch.stack(labels, dim=0)  # (B,)

    return {
        "waveforms": waveforms_tensor,
        "prosodics": prosodics_tensor,
        "labels": labels_tensor,
    }


