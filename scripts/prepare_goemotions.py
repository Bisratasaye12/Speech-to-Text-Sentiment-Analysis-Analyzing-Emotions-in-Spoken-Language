"""
Prepare GoEmotions for 6-way emotion classification compatible with CREMA-D.

Separation of concerns:
- This script:
    * downloads/loads GoEmotions via HuggingFace Datasets
    * maps fine-grained labels → 6 canonical emotions
    * drops ambiguous / multi-label examples
    * writes a clean CSV with columns: [text, label]
- It does NOT handle model training; that will live in a separate script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import sys

import pandas as pd
from datasets import load_dataset

# Ensure project root is on sys.path so we can import `config` and `src.*`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.text_emotion_mapping import map_goemotions_labels, CANONICAL_EMOTIONS


def prepare_goemotions(split: str = "train", output_dir: Path | None = None) -> Path:
    """
    Load GoEmotions, map to 6-way labels, and save a CSV.

    Parameters
    ----------
    split:
        Dataset split to use ("train", "validation", "test", or "train+validation").
    output_dir:
        Directory to write the CSV into. Defaults to `config.RAW_DATA_DIR`.

    Returns
    -------
    Path to the written CSV file.
    """
    if output_dir is None:
        output_dir = config.RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = "go_emotions"
    subset = "raw"  # as configured in config.GOEMOTIONS_CONFIG

    # The 'raw' configuration of go_emotions only has a 'train' split.
    # We emulate additional splits by:
    # - using the full train set when split in {"train", "train+validation"}
    # - and raising for unsupported options.
    if split in {"train", "train+validation"}:
        ds_hf = load_dataset(dataset_name, subset, split="train")
        ds = ds_hf.to_pandas()
        effective_split = "train" if split == "train" else "train_full"
    elif split == "test":
        raise ValueError("The 'raw' config of go_emotions has no separate test split.")
    else:
        raise ValueError(f"Unsupported split '{split}' for go_emotions/raw.")

    print(f"[1/3] Loaded GoEmotions split='{split}' → {len(ds)} examples")

    # In the 'raw' config, each emotion is a separate int column (0/1),
    # e.g. columns: 'anger', 'joy', 'sadness', 'neutral', ...
    # We convert active emotion columns into label names.
    emotion_cols: List[str] = [
        col
        for col in ds.columns
        if col in {
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        }
    ]

    def map_row_labels(row) -> str | None:
        active = [col for col in emotion_cols if row[col] == 1]
        return map_goemotions_labels(active)

    print("[2/3] Mapping GoEmotions labels → canonical 6-way emotions...")
    ds["canonical_label"] = ds.apply(map_row_labels, axis=1)

    before = len(ds)
    ds = ds[ds["canonical_label"].notna()].copy()
    after = len(ds)

    print(f"      Kept {after} examples, dropped {before - after} ambiguous/unsupported.")
    print(f"      Classes: {sorted(CANONICAL_EMOTIONS)}")

    df_out = ds[["text", "canonical_label"]].rename(
        columns={"text": "text", "canonical_label": "label"}
    )

    output_file = output_dir / f"goemotions_6class_{effective_split}.csv"
    df_out.to_csv(output_file, index=False)

    print(f"[3/3] Saved cleaned dataset to: {output_file.relative_to(config.PROJECT_ROOT)}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare GoEmotions for 6-way emotion classification."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train+validation",
        choices=["train", "validation", "test", "train+validation"],
        help="Which GoEmotions split to prepare.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory (defaults to config.RAW_DATA_DIR).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir is not None else None
    prepare_goemotions(split=args.split, output_dir=out_dir)


if __name__ == "__main__":
    main()


