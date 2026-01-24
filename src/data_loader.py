"""
CREMA-D Dataset Loader
Parses filenames and extracts metadata from CREMA-D audio files
"""
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import config


class CREMADataLoader:
    """
    Loads and parses CREMA-D dataset metadata from filenames
    
    Filename format: {ActorID}_{Sentence}_{Emotion}_{Intensity}.wav
    Example: 1001_DFA_ANG_XX.wav
    """
    
    def __init__(self, audio_dir: Path = None):
        """
        Initialize the data loader
        
        Args:
            audio_dir: Path to directory containing CREMA-D audio files
        """
        self.audio_dir = audio_dir or config.AUDIO_WAV_DIR
        self.emotion_mapping = config.EMOTION_MAPPING
        self.intensity_mapping = config.INTENSITY_MAPPING
        
    def parse_filename(self, filename: str) -> Dict:
        """
        Parse CREMA-D filename to extract metadata
        
        Args:
            filename: Audio filename (e.g., '1001_DFA_ANG_XX.wav')
            
        Returns:
            Dictionary with parsed metadata
        """
        # Remove extension
        name = Path(filename).stem
        
        # Pattern: ActorID_Sentence_Emotion_Intensity
        pattern = r'^(\d+)_([A-Z]+)_([A-Z]+)_([A-Z]+)$'
        match = re.match(pattern, name)
        
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        
        actor_id, sentence, emotion_code, intensity_code = match.groups()
        
        # Map codes to full names
        emotion = self.emotion_mapping.get(emotion_code, emotion_code.lower())
        intensity = self.intensity_mapping.get(intensity_code, intensity_code.lower())
        
        return {
            "filename": filename,
            "actor_id": int(actor_id),
            "sentence": sentence,
            "emotion_code": emotion_code,
            "emotion": emotion,
            "intensity_code": intensity_code,
            "intensity": intensity,
            "filepath": str(self.audio_dir / filename)
        }
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load all audio files and create metadata DataFrame
        
        Returns:
            DataFrame with columns: filename, actor_id, sentence, emotion, intensity, filepath
        """
        audio_files = list(self.audio_dir.glob("*.wav"))
        
        if not audio_files:
            raise FileNotFoundError(f"No WAV files found in {self.audio_dir}")
        
        metadata_list = []
        errors = []
        
        for audio_file in audio_files:
            try:
                metadata = self.parse_filename(audio_file.name)
                metadata_list.append(metadata)
            except Exception as e:
                errors.append(f"{audio_file.name}: {str(e)}")
        
        if errors:
            print(f"Warning: {len(errors)} files could not be parsed:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        df = pd.DataFrame(metadata_list)
        
        # Add duration if librosa is available
        try:
            import librosa
            durations = []
            for filepath in df['filepath']:
                try:
                    duration = librosa.get_duration(path=filepath)
                    durations.append(duration)
                except:
                    durations.append(None)
            df['duration'] = durations
        except ImportError:
            print("Warning: librosa not available, skipping duration calculation")
        
        return df
    
    def get_emotion_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get emotion distribution in the dataset
        
        Args:
            df: Metadata DataFrame
            
        Returns:
            DataFrame with emotion counts and percentages
        """
        emotion_counts = df['emotion'].value_counts().sort_index()
        emotion_pct = df['emotion'].value_counts(normalize=True).sort_index() * 100
        
        distribution = pd.DataFrame({
            'count': emotion_counts,
            'percentage': emotion_pct
        })
        
        return distribution
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get dataset statistics
        
        Args:
            df: Metadata DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_files': len(df),
            'unique_actors': df['actor_id'].nunique(),
            'unique_sentences': df['sentence'].nunique(),
            'emotions': df['emotion'].nunique(),
            'emotion_distribution': self.get_emotion_distribution(df).to_dict(),
        }
        
        if 'duration' in df.columns and df['duration'].notna().any():
            stats['avg_duration'] = df['duration'].mean()
            stats['min_duration'] = df['duration'].min()
            stats['max_duration'] = df['duration'].max()
        
        return stats

