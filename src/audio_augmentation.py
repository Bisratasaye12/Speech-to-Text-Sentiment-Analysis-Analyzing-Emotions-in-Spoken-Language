"""
Audio Data Augmentation for Emotion Recognition
"""
import numpy as np
import librosa
import random
from typing import Optional


class AudioAugmentation:
    """Audio augmentation techniques for emotion recognition"""
    
    def __init__(
        self,
        time_shift_prob: float = 0.5,
        pitch_shift_prob: float = 0.5,
        noise_prob: float = 0.3,
        speed_change_prob: float = 0.4,
        time_stretch_prob: float = 0.3,
        volume_change_prob: float = 0.3,
    ):
        self.time_shift_prob = time_shift_prob
        self.pitch_shift_prob = pitch_shift_prob
        self.noise_prob = noise_prob
        self.speed_change_prob = speed_change_prob
        self.time_stretch_prob = time_stretch_prob
        self.volume_change_prob = volume_change_prob
    
    def apply(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Apply random augmentations to audio"""
        augmented = audio.copy()
        
        # Time shift (circular shift)
        if random.random() < self.time_shift_prob:
            shift = random.randint(-int(sr * 0.1), int(sr * 0.1))
            augmented = np.roll(augmented, shift)
        
        # Pitch shift (preserves duration)
        if random.random() < self.pitch_shift_prob:
            n_steps = random.uniform(-2, 2)  # ±2 semitones
            augmented = librosa.effects.pitch_shift(
                augmented, sr=sr, n_steps=n_steps
            )
        
        # Add noise
        if random.random() < self.noise_prob:
            noise_level = random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented = augmented + noise
            augmented = np.clip(augmented, -1.0, 1.0)
        
        # Speed change (time stretch + pitch shift combo)
        if random.random() < self.speed_change_prob:
            rate = random.uniform(0.9, 1.1)  # ±10% speed
            augmented = librosa.effects.time_stretch(augmented, rate=rate)
            # Pad or trim to original length
            target_len = len(audio)
            if len(augmented) < target_len:
                augmented = np.pad(augmented, (0, target_len - len(augmented)))
            else:
                augmented = augmented[:target_len]
        
        # Time stretch (preserves pitch)
        if random.random() < self.time_stretch_prob:
            rate = random.uniform(0.9, 1.1)
            augmented = librosa.effects.time_stretch(augmented, rate=rate)
            target_len = len(audio)
            if len(augmented) < target_len:
                augmented = np.pad(augmented, (0, target_len - len(augmented)))
            else:
                augmented = augmented[:target_len]
        
        # Volume change
        if random.random() < self.volume_change_prob:
            gain = random.uniform(0.7, 1.3)
            augmented = augmented * gain
            augmented = np.clip(augmented, -1.0, 1.0)
        
        return augmented


def apply_augmentation(
    audio: np.ndarray,
    sr: int = 16000,
    augment: bool = True,
    aug_config: Optional[AudioAugmentation] = None,
) -> np.ndarray:
    """Apply augmentation if enabled"""
    if not augment:
        return audio
    
    if aug_config is None:
        aug_config = AudioAugmentation()
    
    return aug_config.apply(audio, sr)

