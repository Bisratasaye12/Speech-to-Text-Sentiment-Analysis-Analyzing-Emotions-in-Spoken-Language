"""
Audio Preprocessing Pipeline for Whisper
Optimized for CREMA-D dataset
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import config


class AudioPreprocessor:
    """
    Preprocesses audio files for Whisper transcription
    Uses minimal preprocessing to preserve audio quality
    """
    
    def __init__(self, target_sr: int = None, normalize: bool = None):
        """
        Initialize preprocessor
        
        Args:
            target_sr: Target sample rate (default: 16000 for Whisper)
            normalize: Whether to normalize audio (default: True)
        """
        self.target_sr = target_sr or config.AUDIO_CONFIG["target_sample_rate"]
        self.normalize = normalize if normalize is not None else config.AUDIO_CONFIG["normalize"]
        self.min_duration = config.AUDIO_CONFIG["min_duration"]
        self.max_duration = config.AUDIO_CONFIG["max_duration"]
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            audio, sr = librosa.load(
                filepath,
                sr=self.target_sr,
                mono=config.AUDIO_CONFIG["mono"]
            )
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio from {filepath}: {str(e)}")
    
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset (mean value) from audio
        
        Args:
            audio: Audio array
            
        Returns:
            Audio with DC offset removed
        """
        return audio - np.mean(audio)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Peak normalization to [-1, 1] range
        
        Args:
            audio: Audio array
            
        Returns:
            Normalized audio array
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio.astype(np.float32)
    
    def validate_duration(self, audio: np.ndarray, sr: int) -> Tuple[bool, float]:
        """
        Validate audio duration is within acceptable range
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Tuple of (is_valid, duration)
        """
        duration = len(audio) / sr
        is_valid = self.min_duration <= duration <= self.max_duration
        return is_valid, duration
    
    def preprocess(self, input_path: str, output_path: Optional[str] = None, 
                   save: bool = False) -> Tuple[np.ndarray, int, dict]:
        """
        Complete preprocessing pipeline
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio (if save=True)
            save: Whether to save processed audio to disk
            
        Returns:
            Tuple of (processed_audio, sample_rate, metadata)
        """
        # Load audio
        audio, sr = self.load_audio(input_path)
        
        # Remove DC offset
        audio = self.remove_dc_offset(audio)
        
        # Normalize (if enabled)
        if self.normalize:
            audio = self.normalize_audio(audio)
        
        # Validate duration
        is_valid, duration = self.validate_duration(audio, sr)
        
        metadata = {
            "original_path": input_path,
            "sample_rate": sr,
            "duration": duration,
            "is_valid": is_valid,
            "original_length": len(audio),
        }
        
        # Save if requested
        if save and output_path:
            if not is_valid:
                raise ValueError(
                    f"Audio duration {duration:.2f}s is outside valid range "
                    f"[{self.min_duration}, {self.max_duration}] seconds"
                )
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as WAV file
            sf.write(output_path, audio, sr, subtype='PCM_16')
            metadata["saved_path"] = output_path
        
        return audio, sr, metadata
    
    def batch_preprocess(self, filepaths: list, output_dir: Path, 
                        verbose: bool = True) -> list:
        """
        Preprocess multiple audio files in batch
        
        Args:
            filepaths: List of input file paths
            output_dir: Directory to save processed files
            verbose: Whether to show progress
            
        Returns:
            List of metadata dictionaries
        """
        from tqdm import tqdm
        
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_list = []
        errors = []
        
        iterator = tqdm(filepaths, desc="Preprocessing") if verbose else filepaths
        
        for filepath in iterator:
            try:
                filename = Path(filepath).name
                output_path = output_dir / filename
                
                _, _, metadata = self.preprocess(
                    str(filepath),
                    str(output_path),
                    save=True
                )
                metadata_list.append(metadata)
                
            except Exception as e:
                errors.append({
                    "filepath": str(filepath),
                    "error": str(e)
                })
        
        if errors and verbose:
            print(f"\nWarning: {len(errors)} files failed preprocessing:")
            for error in errors[:10]:
                print(f"  - {error['filepath']}: {error['error']}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        return metadata_list

