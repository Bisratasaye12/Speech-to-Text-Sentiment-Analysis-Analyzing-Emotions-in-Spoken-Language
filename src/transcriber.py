"""
Whisper Transcription Module
Transcribes audio files using OpenAI Whisper
"""
import warnings
import whisper
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import config

# Suppress Whisper warnings about FP16 on CPU
warnings.filterwarnings('ignore', category=UserWarning, module='whisper')


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
            print(f"Loading Whisper model: {self.model_name} on {self.device}...", end=" ", flush=True)
            # Suppress warnings during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.model = whisper.load_model(self.model_name, device=self.device)
            print("✓")
    
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
        # Suppress warnings during transcription
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            result = self.model.transcribe(
                audio_path,
                language=language,
                verbose=verbose,
                task="transcribe",
                fp16=False  # Explicitly disable FP16 to avoid warnings on CPU
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
                            show_progress: bool = True,
                            skip_existing: bool = True,
                            save_checkpoint: Optional[str] = None,
                            checkpoint_interval: int = 50) -> pd.DataFrame:
        """
        Transcribe audio files from a DataFrame with checkpoint/resume support
        
        Args:
            df: DataFrame with audio file paths
            audio_path_col: Column name containing audio file paths
            output_col: Column name to store transcriptions
            language: Language code (default: 'en')
            show_progress: Whether to show progress bar
            skip_existing: Skip files that already have transcriptions
            save_checkpoint: Path to save checkpoint file (optional)
            checkpoint_interval: Save checkpoint every N files (default: 50)
            
        Returns:
            DataFrame with added transcription column
        """
        if self.model is None:
            self.load_model()
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Initialize output columns if they don't exist
        if output_col not in df.columns:
            df[output_col] = ''
        if f'{output_col}_language' not in df.columns:
            df[f'{output_col}_language'] = ''
        if f'{output_col}_no_speech_prob' not in df.columns:
            df[f'{output_col}_no_speech_prob'] = 0.0
        if f'{output_col}_success' not in df.columns:
            df[f'{output_col}_success'] = False
        
        # Filter out rows with missing audio paths
        valid_df = df[df[audio_path_col].notna()].copy()
        
        if len(valid_df) == 0:
            print("Warning: No valid audio paths found in DataFrame")
            return df
        
        # Skip already transcribed files if requested
        if skip_existing:
            # Check which files need transcription
            needs_transcription = (
                (valid_df[output_col].isna()) | 
                (valid_df[output_col] == '') |
                (valid_df[f'{output_col}_success'] == False)
            )
            valid_df = valid_df[needs_transcription].copy()
            
            already_done = len(df) - len(valid_df)
            if already_done > 0:
                print(f"  ℹ Skipping {already_done} already transcribed files")
        
        if len(valid_df) == 0:
            print("  ✓ All files already transcribed!")
            return df
        
        # Get audio paths and their indices
        audio_paths = valid_df[audio_path_col].tolist()
        audio_indices = valid_df.index.tolist()
        
        # Transcribe with checkpoint support
        transcriptions = []
        iterator = tqdm(enumerate(audio_paths), total=len(audio_paths), desc="Transcribing") if show_progress else enumerate(audio_paths)
        
        for idx, audio_path in iterator:
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
            
            # Save checkpoint periodically
            if save_checkpoint and (idx + 1) % checkpoint_interval == 0:
                # Update DataFrame with current progress
                for i, trans in enumerate(transcriptions):
                    orig_idx = audio_indices[i]
                    df.loc[orig_idx, output_col] = trans['text']
                    df.loc[orig_idx, f'{output_col}_language'] = trans['language']
                    df.loc[orig_idx, f'{output_col}_no_speech_prob'] = trans['no_speech_prob']
                    df.loc[orig_idx, f'{output_col}_success'] = trans['success']
                
                # Save checkpoint
                df.to_csv(save_checkpoint, index=False)
                if show_progress:
                    iterator.set_postfix({"checkpoint": "saved"})
        
        # Update DataFrame with all transcriptions
        for i, trans in enumerate(transcriptions):
            orig_idx = audio_indices[i]
            df.loc[orig_idx, output_col] = trans['text']
            df.loc[orig_idx, f'{output_col}_language'] = trans['language']
            df.loc[orig_idx, f'{output_col}_no_speech_prob'] = trans['no_speech_prob']
            df.loc[orig_idx, f'{output_col}_success'] = trans['success']
        
        return df

