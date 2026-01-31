"""
Multi-Dataset Loader for Audio Emotion Recognition
Supports: CREMA-D, TESS, IEMOCAP
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import re

import config


class TESSDataLoader:
    """
    Load TESS (Toronto Emotional Speech Set) dataset metadata.
    
    TESS filename formats:
    - Actor_Emotion_Sentence.wav (e.g., OAF_angry_01.wav)
    - Actor_Emotion.wav (e.g., OAF_angry.wav)
    """
    
    # TESS emotion mapping to canonical 6-way emotions
    TESS_EMOTION_MAPPING = {
        "angry": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "surprise": "neutral",  # Map surprise to neutral (or could be separate)
        "ps": "neutral",  # Pleasant surprise → neutral
    }
    
    def __init__(self, audio_dir: Path):
        """
        Initialize TESS data loader.
        
        Args:
            audio_dir: Path to directory containing TESS audio files
        """
        self.audio_dir = Path(audio_dir)
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"TESS directory not found: {audio_dir}")
    
    def parse_filename(self, filename: str) -> Dict:
        """
        Parse TESS filename to extract emotion and metadata.
        
        Supports multiple TESS filename formats:
        - YAF_date_disgust.wav (word_emotion format)
        - OAF_angry_01.wav (actor_emotion_sentence format)
        - OAF_angry.wav (actor_emotion format)
        
        Args:
            filename: Audio filename
            
        Returns:
            Dictionary with parsed metadata
        """
        name = Path(filename).stem
        
        # Pattern 1: Actor_Emotion_Sentence (e.g., OAF_angry_01)
        pattern1 = r'^([A-Z]+)_([a-z]+)_(\d+)$'
        match1 = re.match(pattern1, name)
        
        if match1:
            actor, emotion_raw, sentence_num = match1.groups()
            emotion = self.TESS_EMOTION_MAPPING.get(emotion_raw.lower(), "neutral")
            
            return {
                "filename": filename,
                "actor_id": actor,
                "emotion": emotion,
                "emotion_raw": emotion_raw,
                "sentence": sentence_num,
                "filepath": str(self.audio_dir / filename),
                "dataset": "TESS",
            }
        
        # Pattern 2: Actor_Emotion (e.g., OAF_angry)
        pattern2 = r'^([A-Z]+)_([a-z]+)$'
        match2 = re.match(pattern2, name)
        
        if match2:
            actor, emotion_raw = match2.groups()
            emotion = self.TESS_EMOTION_MAPPING.get(emotion_raw.lower(), "neutral")
            
            return {
                "filename": filename,
                "actor_id": actor,
                "emotion": emotion,
                "emotion_raw": emotion_raw,
                "sentence": None,
                "filepath": str(self.audio_dir / filename),
                "dataset": "TESS",
            }
        
        # Pattern 3: Actor_Word_Emotion (e.g., YAF_date_disgust)
        # This is common in nested folder structure
        pattern3 = r'^([A-Z]+)_([a-z]+)_([a-z_]+)$'
        match3 = re.match(pattern3, name)
        
        if match3:
            actor, word, emotion_raw = match3.groups()
            emotion = self.TESS_EMOTION_MAPPING.get(emotion_raw.lower(), "neutral")
            
            return {
                "filename": filename,
                "actor_id": actor,
                "emotion": emotion,
                "emotion_raw": emotion_raw,
                "sentence": word,  # Use word as sentence identifier
                "filepath": str(self.audio_dir / filename),
                "dataset": "TESS",
            }
        
        raise ValueError(f"Could not parse TESS filename: {filename}")
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load all TESS files and create metadata DataFrame.
        
        Supports both flat structure and nested folder structure.
        
        Returns:
            DataFrame with columns: filename, actor_id, emotion, filepath, dataset
        """
        # Try to find WAV files - check nested structure first
        audio_files = []
        
        # Check for nested structure: TESS Toronto emotional speech set data/
        nested_dir = self.audio_dir / "TESS Toronto emotional speech set data"
        if nested_dir.exists():
            # Find all WAV files in subdirectories
            audio_files = list(nested_dir.rglob("*.wav"))
        else:
            # Try flat structure or direct subdirectories
            audio_files = list(self.audio_dir.rglob("*.wav"))
        
        if not audio_files:
            raise FileNotFoundError(f"No WAV files found in {self.audio_dir}")
        
        metadata = []
        errors = []
        
        for file in audio_files:
            try:
                # Parse filename - handle both formats
                # Format 1: YAF_date_disgust.wav (in YAF_disgust folder)
                # Format 2: OAF_angry_01.wav (flat)
                meta = self.parse_filename(file.name)
                # Update filepath to actual location
                meta["filepath"] = str(file)
                metadata.append(meta)
            except ValueError as e:
                errors.append(f"{file.name}: {str(e)}")
        
        if errors:
            print(f"Warning: {len(errors)} TESS files could not be parsed:")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        df = pd.DataFrame(metadata)
        
        if len(df) == 0:
            raise ValueError(f"No valid TESS files found in {self.audio_dir}")
        
        return df


class IEMOCAPDataLoader:
    """
    Load IEMOCAP (Interactive Emotional Dyadic Motion Capture) dataset metadata.
    
    IEMOCAP structure:
    - Session1/Sentences/wav/SpeakerID/...wav
    - Session1/EmoEvaluation/SpeakerID.txt (annotations)
    """
    
    # IEMOCAP emotion mapping to canonical 6-way emotions
    IEMOCAP_EMOTION_MAPPING = {
        "ang": "anger",
        "hap": "happy",
        "exc": "happy",  # Excitement → happy
        "sad": "sad",
        "neu": "neutral",
        "fru": "anger",  # Frustration → anger
        "fea": "fear",
        "dis": "disgust",
        "sur": "neutral",  # Surprise → neutral
        "xxx": None,  # Invalid/unknown
    }
    
    def __init__(self, audio_dir: Path):
        """
        Initialize IEMOCAP data loader.
        
        Args:
            audio_dir: Path to IEMOCAP root directory (contains Session1/, Session2/, etc.)
        """
        self.audio_dir = Path(audio_dir)
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"IEMOCAP directory not found: {audio_dir}")
    
    def load_annotations(self, session_dir: Path) -> Dict[str, str]:
        """
        Load emotion annotations from EmoEvaluation files.
        
        Args:
            session_dir: Path to session directory (e.g., Session1/)
            
        Returns:
            Dictionary mapping filename to emotion label
        """
        annotations = {}
        # Try different possible locations for EmoEvaluation
        emo_eval_dir = session_dir / "EmoEvaluation"
        if not emo_eval_dir.exists():
            emo_eval_dir = session_dir / "dialog" / "EmoEvaluation"  # Alternative structure
        if not emo_eval_dir.exists():
            emo_eval_dir = session_dir / "Dialog" / "EmoEvaluation"  # Capitalized
        if not emo_eval_dir.exists():
            print(f"Warning: EmoEvaluation directory not found in {session_dir}")
            return annotations
        
        # Each session has annotation files like: Ses01F_impro01.txt, Ses01F_script01.txt
        for txt_file in emo_eval_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('[') or line.startswith(';'):
                            continue
                        
                        # Parse annotation line
                        # Format varies, but typically: [filename] [emotion] [valence] [arousal] [dominance]
                        # Or: filename emotion ...
                        parts = line.split()
                        if len(parts) >= 2:
                            # Remove brackets if present
                            filename = parts[0].strip('[]')
                            emotion_code = parts[1].lower().strip('[]')
                            emotion = self.IEMOCAP_EMOTION_MAPPING.get(emotion_code)
                            
                            if emotion:  # Only add if emotion is valid (not None)
                                annotations[filename] = emotion
            except Exception as e:
                print(f"Warning: Could not parse annotation file {txt_file}: {e}")
        
        return annotations
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load all IEMOCAP files and create metadata DataFrame.
        
        Returns:
            DataFrame with columns: filename, actor_id, emotion, session, filepath, dataset
        """
        metadata = []
        
        # Iterate through sessions (Session1, Session2, ..., Session5)
        session_dirs = sorted([d for d in self.audio_dir.iterdir() if d.is_dir() and d.name.startswith("Session")])
        
        if not session_dirs:
            raise FileNotFoundError(f"No Session directories found in {self.audio_dir}")
        
        for session_dir in session_dirs:
            session_num = session_dir.name
            
            # Load annotations for this session
            # Check both dialog/EmoEvaluation and EmoEvaluation locations
            annotations = self.load_annotations(session_dir)
            
            # Find audio files in Sentences/wav/
            sentences_dir = session_dir / "Sentences" / "wav"
            if not sentences_dir.exists():
                # Try alternative structure: SessionX/wav/
                sentences_dir = session_dir / "wav"
                if not sentences_dir.exists():
                    print(f"Warning: No audio directory found in {session_dir}")
                    continue
            
            # Each speaker has their own folder
            for speaker_dir in sentences_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                
                speaker_id = speaker_dir.name
                
                # Find all wav files
                for wav_file in speaker_dir.glob("*.wav"):
                    filename = wav_file.name
                    # Try to get emotion from annotations, default to neutral
                    emotion = annotations.get(filename, "neutral")
                    
                    if emotion:  # Skip if emotion is None
                        metadata.append({
                            "filename": filename,
                            "actor_id": speaker_id,
                            "emotion": emotion,
                            "session": session_num,
                            "filepath": str(wav_file),
                            "dataset": "IEMOCAP",
                        })
        
        if not metadata:
            raise ValueError(f"No valid IEMOCAP files found in {self.audio_dir}")
        
        df = pd.DataFrame(metadata)
        return df


class MultiDatasetLoader:
    """
    Unified loader for multiple emotion datasets (CREMA-D, TESS, IEMOCAP).
    """
    
    def __init__(
        self,
        crema_dir: Optional[Path] = None,
        tess_dir: Optional[Path] = None,
        iemocap_dir: Optional[Path] = None,
    ):
        """
        Initialize multi-dataset loader.
        
        Args:
            crema_dir: Path to CREMA-D audio directory
            tess_dir: Path to TESS audio directory
            iemocap_dir: Path to IEMOCAP root directory
        """
        self.crema_dir = crema_dir
        self.tess_dir = tess_dir
        self.iemocap_dir = iemocap_dir
    
    def load_all_datasets(self) -> pd.DataFrame:
        """
        Load and combine all available datasets.
        
        Returns:
            Combined DataFrame with all datasets, filtered to canonical emotions
        """
        all_dfs = []
        
        # Load CREMA-D
        if self.crema_dir and Path(self.crema_dir).exists():
            try:
                from src.data_loader import CREMADataLoader
                crema_loader = CREMADataLoader(audio_dir=self.crema_dir)
                crema_df = crema_loader.load_metadata()
                crema_df["dataset"] = "CREMA-D"
                all_dfs.append(crema_df)
                print(f"✓ Loaded CREMA-D: {len(crema_df)} files")
            except Exception as e:
                print(f"Warning: Could not load CREMA-D: {e}")
        else:
            print("⚠ CREMA-D directory not found or not specified")
        
        # Load TESS
        if self.tess_dir and Path(self.tess_dir).exists():
            try:
                tess_loader = TESSDataLoader(audio_dir=self.tess_dir)
                tess_df = tess_loader.load_metadata()
                all_dfs.append(tess_df)
                print(f"✓ Loaded TESS: {len(tess_df)} files")
            except Exception as e:
                print(f"Warning: Could not load TESS: {e}")
        else:
            print("⚠ TESS directory not found or not specified")
        
        # Load IEMOCAP
        if self.iemocap_dir and Path(self.iemocap_dir).exists():
            try:
                iemocap_loader = IEMOCAPDataLoader(audio_dir=self.iemocap_dir)
                iemocap_df = iemocap_loader.load_metadata()
                all_dfs.append(iemocap_df)
                print(f"✓ Loaded IEMOCAP: {len(iemocap_df)} files")
            except Exception as e:
                print(f"Warning: Could not load IEMOCAP: {e}")
        else:
            print("⚠ IEMOCAP directory not found or not specified")
        
        if not all_dfs:
            raise ValueError("No datasets found! Please provide at least one dataset directory.")
        
        # Combine all DataFrames
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Filter to canonical emotions only
        canonical_emotions = set(config.CANONICAL_EMOTIONS)
        before_count = len(combined_df)
        combined_df = combined_df[combined_df["emotion"].isin(canonical_emotions)]
        after_count = len(combined_df)
        
        if before_count != after_count:
            print(f"⚠ Filtered {before_count - after_count} files with non-canonical emotions")
        
        print(f"\n✓ Total combined dataset: {len(combined_df)} files")
        print(f"  Emotions: {combined_df['emotion'].value_counts().to_dict()}")
        print(f"  Datasets: {combined_df['dataset'].value_counts().to_dict()}")
        
        return combined_df


__all__ = [
    "TESSDataLoader",
    "IEMOCAPDataLoader",
    "MultiDatasetLoader",
]

