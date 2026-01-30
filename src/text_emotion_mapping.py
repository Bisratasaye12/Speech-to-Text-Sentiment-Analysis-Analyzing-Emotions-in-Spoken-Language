"""
Utilities for mapping external text emotion datasets (e.g. GoEmotions)
to the 6 canonical emotions used in this project:
    anger, disgust, fear, happy, neutral, sad

Separation of concerns:
- This module only knows about LABEL MAPPING and simple filtering logic.
- Dataset loading, model training, etc. live in their own modules/scripts.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set

import config


# Fine-grained GoEmotions labels → canonical 6-way emotions
GOEMOTIONS_TO_CANONICAL = {
    # anger
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    # disgust
    "disgust": "disgust",
    # fear
    "fear": "fear",
    "nervousness": "fear",
    # happy
    "joy": "happy",
    "amusement": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "love": "happy",
    "optimism": "happy",
    "relief": "happy",
    "pride": "happy",
    "admiration": "happy",
    # sad
    "sadness": "sad",
    "disappointment": "sad",
    "embarrassment": "sad",
    "grief": "sad",
    "remorse": "sad",
    # neutral
    "neutral": "neutral",
}


CANONICAL_EMOTIONS: Set[str] = set(config.CANONICAL_EMOTIONS)


def map_goemotions_labels(labels: Iterable[str]) -> Optional[str]:
    """
    Map a set of GoEmotions labels for a single example to ONE canonical emotion.

    Strategy:
    - Map each fine-grained GoEmotions label to its canonical emotion (if supported)
    - Collect the set of canonical emotions present
    - If exactly ONE canonical emotion is present → return it
    - Otherwise (0 or >1) → return None (example is ambiguous or unsupported)

    This is conservative but helps avoid noisy supervision when fine-tuning.
    """
    mapped: Set[str] = set()
    for lbl in labels:
        lbl = lbl.strip().lower()
        if lbl in GOEMOTIONS_TO_CANONICAL:
            mapped.add(GOEMOTIONS_TO_CANONICAL[lbl])

    if len(mapped) == 1:
        # Unambiguous mapping
        return next(iter(mapped))

    # Either no supported emotion or multiple canonical emotions → skip
    return None


__all__ = [
    "GOEMOTIONS_TO_CANONICAL",
    "CANONICAL_EMOTIONS",
    "map_goemotions_labels",
]


