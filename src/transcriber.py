"""
Whisper Transcription Module
Transcribes audio files using OpenAI Whisper
"""
import whisper
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import config


class WhisperTranscriber:
    """
    Transcribes audio files using OpenAI Whisper
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.model_name = model_name or config.WHISPER_MODEL
        self.device = self._get_device(device)
        self.model = None
        
    def _get_device(self, device: Optional[str] = None) -> str:
        """
        Determine the best device to use
        
        Args:
            device: Optional device specification
            
        Returns:
            Device string ('cpu', 'cuda', or 'mps')
        """
        if device:
            return device
        
        # Whisper has better support for CUDA, CPU is more stable than MPS
        if torch.cuda.is_available():
            return "cuda"
        # Note: MPS support in Whisper is limited, use CPU as fallback
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     return "mps"
        else:
            return "cpu"
    
    def load_model(self):
        """
        Load Whisper model
        """
        if self.model is None:
            print(f"Loading Whisper model: {self.model_name} on {self.device}...")
            self.model = whisper.load_model(self.model_name, device=self.device)
            print(f"✓ Model loaded successfully")
    
    def transcribe(self, audio_path: str, language: str = "en", 
                   verbose: bool = False) -> Dict:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: 'en' for English)
            verbose: Whether to print transcription details
            
        Returns:
            Dictionary with transcription results:
            {
                'text': str,           # Transcribed text
                'language': str,        # Detected language
                'segments': list,      # Word-level segments (if available)
                'no_speech_prob': float # Probability of no speech
            }
        """
        if self.model is None:
            self.load_model()
        
        # Load and transcribe
        result = self.model.transcribe(
            audio_path,
            language=language,
            verbose=verbose,
            task="transcribe"
        )
        
        return {
            'text': result['text'].strip(),
            'language': result.get('language', language),
            'segments': result.get('segments', []),
            'no_speech_prob': result.get('no_speech_prob', 0.0)
        }
    
    def transcribe_batch(self, audio_paths: List[str], language: str = "en",
                        show_progress: bool = True) -> List[Dict]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            language: Language code (default: 'en')
            show_progress: Whether to show progress bar
            
        Returns:
            List of transcription dictionaries
        """
        if self.model is None:
            self.load_model()
        
        transcriptions = []
        iterator = tqdm(audio_paths, desc="Transcribing") if show_progress else audio_paths
        
        for audio_path in iterator:
            try:
                # Check if file exists
                if not Path(audio_path).exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
                result = self.transcribe(audio_path, language=language, verbose=False)
                result['audio_path'] = audio_path
                result['success'] = True
            except Exception as e:
                result = {
                    'audio_path': audio_path,
                    'text': '',
                    'language': language,
                    'segments': [],
                    'no_speech_prob': 1.0,
                    'success': False,
                    'error': str(e)
                }
                # Print error for debugging (only first few)
                if len(transcriptions) < 3:
                    print(f"\n  Warning: Failed to transcribe {Path(audio_path).name}: {e}")
            transcriptions.append(result)
        
        return transcriptions
    
    def transcribe_dataframe(self, df: pd.DataFrame, 
                            audio_path_col: str = 'processed_filepath',
                            output_col: str = 'transcription',
                            language: str = "en",
                            show_progress: bool = True) -> pd.DataFrame:
        """
        Transcribe audio files from a DataFrame
        
        Args:
            df: DataFrame with audio file paths
            audio_path_col: Column name containing audio file paths
            output_col: Column name to store transcriptions
            language: Language code (default: 'en')
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with added transcription column
        """
        if self.model is None:
            self.load_model()
        
        # Filter out rows with missing audio paths
        valid_df = df[df[audio_path_col].notna()].copy()
        
        if len(valid_df) == 0:
            print("Warning: No valid audio paths found in DataFrame")
            df[output_col] = ''
            return df
        
        # Get audio paths
        audio_paths = valid_df[audio_path_col].tolist()
        
        # Transcribe
        transcriptions = self.transcribe_batch(
            audio_paths, 
            language=language,
            show_progress=show_progress
        )
        
        # Extract text from transcriptions
        texts = [t['text'] for t in transcriptions]
        
        # Add to DataFrame
        df = df.copy()
        df[output_col] = ''
        df.loc[valid_df.index, output_col] = texts
        
        # Add additional metadata columns
        df[f'{output_col}_language'] = ''
        df.loc[valid_df.index, f'{output_col}_language'] = [t['language'] for t in transcriptions]
        
        df[f'{output_col}_no_speech_prob'] = 0.0
        df.loc[valid_df.index, f'{output_col}_no_speech_prob'] = [t['no_speech_prob'] for t in transcriptions]
        
        df[f'{output_col}_success'] = False
        df.loc[valid_df.index, f'{output_col}_success'] = [t['success'] for t in transcriptions]
        
        return df

