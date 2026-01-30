"""
Evaluate Fused Multimodal Emotion Model
---------------------------------------

This script evaluates:
1. Audio-only model
2. Text-only model  
3. Fused model (audio + text)

Compares accuracy, F1 scores, and per-class performance.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

import config
from src.audio_emotion_dataset import CREMAAudioEmotionDataset, collate_fn
from src.audio_emotion_model import AudioEmotionConfig, DistilHuBERTEmotionModel, AudioEmotionInference
from src.text_model import load_roberta_text_model
from src.fusion_model import FusionConfig, MultimodalFusionModel, MultimodalEmotionInference
from src.transcriber import WhisperTranscriber
from src.audio_features import extract_prosodic_features, prosodic_feature_vector
from src.preprocessor import AudioPreprocessor


def get_device() -> torch.device:
    """Get the best available device"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_models(device: torch.device):
    """Load all models needed for evaluation"""
    print("=" * 70)
    print("Loading Models")
    print("=" * 70)
    
    # Load audio model
    print("\n1. Loading audio emotion model...")
    checkpoint_paths = [
        config.PROCESSED_DATA_DIR / "audio_emotion_model" / "best_model.pt",
        config.PROCESSED_DATA_DIR / "audio_emotion_model_improved" / "best_model.pt",
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if path.exists():
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError("No trained audio model checkpoint found")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg_dict = checkpoint["config"]
    cfg = AudioEmotionConfig(**cfg_dict)
    model = DistilHuBERTEmotionModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    audio_id2label = checkpoint["id2label"]
    audio_model = AudioEmotionInference(model, audio_id2label, device=str(device))
    print(f"   ✓ Audio model loaded: {checkpoint_path.name}")
    print(f"   Labels: {[audio_id2label[i] for i in sorted(audio_id2label.keys())]}")
    
    # Load text model
    print("\n2. Loading text emotion model...")
    text_checkpoint_paths = [
        config.PROCESSED_DATA_DIR / "text_model" / "best_model.pt",
        config.PROCESSED_DATA_DIR / "models" / "checkpoints" / "best_model.pt",
    ]
    
    text_checkpoint_path = None
    for path in text_checkpoint_paths:
        if path.exists():
            text_checkpoint_path = path
            break
    
    text_artifacts = load_roberta_text_model(
        model_name="roberta-base",
        use_lora=True,
        freeze_base=True,
    )
    
    if text_checkpoint_path is not None:
        print(f"   Loading from: {text_checkpoint_path}")
        # Load on CPU first to avoid MPS alignment issues
        checkpoint = torch.load(text_checkpoint_path, map_location="cpu")
        try:
            text_artifacts.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"   ✓ Text model loaded")
        except Exception as e:
            print(f"   ⚠️ Warning loading weights: {e}")
            print(f"   Attempting partial load...")
            # Try loading compatible keys only
            state_dict = checkpoint["model_state_dict"]
            model_dict = text_artifacts.model.state_dict()
            compatible = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(compatible)
            text_artifacts.model.load_state_dict(model_dict, strict=False)
            print(f"   ✓ Text model loaded ({len(compatible)} keys)")
        
        if "id2label" in checkpoint:
            text_artifacts.id2label = checkpoint["id2label"]
    else:
        print(f"   ⚠️ Using pre-trained RoBERTa (not fine-tuned)")
    
    text_artifacts.model.to(device)
    text_artifacts.model.eval()
    print(f"   Labels: {[text_artifacts.id2label[i] for i in sorted(text_artifacts.id2label.keys())]}")
    
    # Load Whisper for transcription
    print("\n3. Loading Whisper transcriber...")
    whisper = WhisperTranscriber(model_name="base")
    whisper.load_model()
    print("   ✓ Whisper loaded")
    
    # Load fusion model
    print("\n4. Loading fusion model...")
    audio_emb_dim = audio_model.model.cfg.hidden_dim
    fusion_cfg = FusionConfig(
        audio_embedding_dim=audio_emb_dim,
        text_embedding_dim=config.FUSION_CONFIG["text_embedding_dim"],
        num_labels=len(config.CANONICAL_EMOTIONS),
        fusion_hidden_dim=config.FUSION_CONFIG["fusion_hidden_dim"],
        dropout=config.FUSION_CONFIG["dropout"],
        fusion_type=config.FUSION_CONFIG["fusion_type"],
    )
    
    fusion_model = MultimodalFusionModel(fusion_cfg)
    fusion_model.to(device)
    fusion_model.eval()
    
    canonical_id2label = {i: label for i, label in enumerate(config.CANONICAL_EMOTIONS)}
    multimodal_inf = MultimodalEmotionInference(
        audio_model=audio_model,
        text_model=text_artifacts.model,
        fusion_model=fusion_model,
        id2label=canonical_id2label,
        device=str(device),
        audio_id2label=audio_id2label,
        text_id2label=text_artifacts.id2label,
    )
    print(f"   ✓ Fusion model loaded (type: {config.FUSION_CONFIG['fusion_type']})")
    
    # Load preprocessor
    preprocessor = AudioPreprocessor()
    
    print("\n" + "=" * 70)
    return {
        'audio_model': audio_model,
        'text_artifacts': text_artifacts,
        'multimodal_inf': multimodal_inf,
        'whisper': whisper,
        'preprocessor': preprocessor,
        'audio_id2label': audio_id2label,
        'text_id2label': text_artifacts.id2label,
        'canonical_id2label': canonical_id2label,
    }


def normalize_label_mapping(id2label, canonical_order=None):
    """Create remapping from model's label order to canonical order"""
    if canonical_order is None:
        canonical_order = config.CANONICAL_EMOTIONS
    
    label_to_canonical_idx = {label: idx for idx, label in enumerate(canonical_order)}
    remap = {}
    for model_idx, label in id2label.items():
        if label in label_to_canonical_idx:
            remap[model_idx] = label_to_canonical_idx[label]
    return remap


def remap_probabilities(probs, remap_dict, num_labels):
    """Remap probabilities from model's label order to canonical order"""
    remapped = np.zeros(num_labels)
    for model_idx, canonical_idx in remap_dict.items():
        if model_idx < len(probs):
            remapped[canonical_idx] = probs[model_idx]
    return remapped


@torch.no_grad()
def evaluate_models(models, test_csv_path: Path, max_samples: int = None):
    """Evaluate all models on test set"""
    device = get_device()
    
    # Load test data
    print(f"\nLoading test data from: {test_csv_path}")
    df = pd.read_csv(test_csv_path)
    
    if max_samples:
        df = df.head(max_samples)
        print(f"  Limited to {max_samples} samples for faster evaluation")
    
    print(f"  Total test samples: {len(df)}")
    
    # Get label mappings
    audio_id2label = models['audio_id2label']
    text_id2label = models['text_id2label']
    canonical_id2label = models['canonical_id2label']
    
    # Create remapping functions
    audio_remap = normalize_label_mapping(audio_id2label)
    text_remap = normalize_label_mapping(text_id2label)
    
    # Initialize results
    all_true_labels = []
    all_audio_preds = []
    all_text_preds = []
    all_fused_preds = []
    
    all_audio_probs = []
    all_text_probs = []
    all_fused_probs = []
    
    print("\n" + "=" * 70)
    print("Evaluating on Test Set")
    print("=" * 70)
    
    # Process each sample
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            # Get ground truth label
            emotion = row['emotion']
            if emotion not in config.CANONICAL_EMOTIONS:
                continue
            
            true_label_idx = config.CANONICAL_EMOTIONS.index(emotion)
            all_true_labels.append(true_label_idx)
            
            # Load audio file
            audio_path = None
            for col in ['processed_filepath', 'filepath']:
                if col in row and pd.notna(row[col]):
                    audio_path = Path(row[col])
                    if not audio_path.is_absolute():
                        audio_path = config.PROCESSED_DATA_DIR / audio_path
                    if audio_path.exists():
                        break
            
            if audio_path is None or not audio_path.exists():
                print(f"  ⚠️ Skipping {row.get('filename', idx)}: audio file not found")
                continue
            
            # Load and preprocess audio
            audio, sr = models['preprocessor'].load_audio(str(audio_path))
            if models['preprocessor'].normalize:
                audio = models['preprocessor'].normalize_audio(audio)
            
            # Extract prosodic features
            prosodic_dict = extract_prosodic_features(audio, sr)
            if "prosodic_feature_vector" in prosodic_dict:
                prosodic_vec = prosodic_dict["prosodic_feature_vector"]
            else:
                prosodic_vec = prosodic_feature_vector(prosodic_dict)
            
            # Convert to tensors
            waveform_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            prosodic_tensor = torch.tensor(prosodic_vec, dtype=torch.float32).unsqueeze(0)
            
            # Transcribe audio
            transcription = models['whisper'].transcribe(str(audio_path), language="en")['text']
            
            # Run predictions
            multimodal_results = models['multimodal_inf'].predict(
                waveforms=waveform_tensor,
                prosodic_feats=prosodic_tensor,
                text=transcription if transcription.strip() else None,
                text_tokenizer=models['text_artifacts'].tokenizer,
                return_individual=True,
            )
            
            # Extract and remap predictions
            fused_probs = multimodal_results['fused_probs'][0].cpu().numpy()
            fused_pred_idx = np.argmax(fused_probs)
            all_fused_preds.append(fused_pred_idx)
            all_fused_probs.append(fused_probs)
            
            audio_probs_raw = multimodal_results['audio_probs'][0].cpu().numpy()
            audio_probs = remap_probabilities(audio_probs_raw, audio_remap, len(config.CANONICAL_EMOTIONS))
            audio_pred_idx = np.argmax(audio_probs)
            all_audio_preds.append(audio_pred_idx)
            all_audio_probs.append(audio_probs)
            
            if 'text_probs' in multimodal_results:
                text_probs_raw = multimodal_results['text_probs'][0].cpu().numpy()
                text_probs = remap_probabilities(text_probs_raw, text_remap, len(config.CANONICAL_EMOTIONS))
                text_pred_idx = np.argmax(text_probs)
                all_text_preds.append(text_pred_idx)
                all_text_probs.append(text_probs)
            else:
                # No text prediction available
                all_text_preds.append(-1)  # Mark as unavailable
                all_text_probs.append(np.zeros(len(config.CANONICAL_EMOTIONS)))
        
        except Exception as e:
            print(f"  ⚠️ Error processing sample {idx}: {e}")
            continue
    
    # Convert to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_audio_preds = np.array(all_audio_preds)
    all_fused_preds = np.array(all_fused_preds)
    
    # Filter out samples where text prediction wasn't available
    text_available = np.array(all_text_preds) != -1
    all_text_preds = np.array(all_text_preds)
    
    print(f"\n✓ Processed {len(all_true_labels)} samples")
    print(f"  Text predictions available: {text_available.sum()}/{len(all_true_labels)}")
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    
    results = {}
    
    # Audio-only results
    audio_acc = accuracy_score(all_true_labels, all_audio_preds)
    audio_f1_macro = f1_score(all_true_labels, all_audio_preds, average='macro')
    audio_f1_weighted = f1_score(all_true_labels, all_audio_preds, average='weighted')
    
    print(f"\n1. Audio-Only Model:")
    print(f"   Accuracy:  {audio_acc:.4f} ({audio_acc*100:.2f}%)")
    print(f"   Macro F1:  {audio_f1_macro:.4f}")
    print(f"   Weighted F1: {audio_f1_weighted:.4f}")
    results['audio'] = {
        'accuracy': audio_acc,
        'macro_f1': audio_f1_macro,
        'weighted_f1': audio_f1_weighted,
        'predictions': all_audio_preds,
        'probabilities': all_audio_probs,
    }
    
    # Text-only results (only for samples with text)
    if text_available.sum() > 0:
        text_true = all_true_labels[text_available]
        text_pred = all_text_preds[text_available]
        
        text_acc = accuracy_score(text_true, text_pred)
        text_f1_macro = f1_score(text_true, text_pred, average='macro')
        text_f1_weighted = f1_score(text_true, text_pred, average='weighted')
        
        print(f"\n2. Text-Only Model:")
        print(f"   Accuracy:  {text_acc:.4f} ({text_acc*100:.2f}%)")
        print(f"   Macro F1:  {text_f1_macro:.4f}")
        print(f"   Weighted F1: {text_f1_weighted:.4f}")
        print(f"   (Evaluated on {text_available.sum()} samples with text)")
        results['text'] = {
            'accuracy': text_acc,
            'macro_f1': text_f1_macro,
            'weighted_f1': text_f1_weighted,
            'predictions': text_pred,
            'probabilities': [all_text_probs[i] for i in range(len(all_text_probs)) if text_available[i]],
            'n_samples': text_available.sum(),
        }
    else:
        print(f"\n2. Text-Only Model: No text predictions available")
        results['text'] = None
    
    # Fused model results
    fused_acc = accuracy_score(all_true_labels, all_fused_preds)
    fused_f1_macro = f1_score(all_true_labels, all_fused_preds, average='macro')
    fused_f1_weighted = f1_score(all_true_labels, all_fused_preds, average='weighted')
    
    print(f"\n3. Fused Model ({config.FUSION_CONFIG['fusion_type']} fusion):")
    print(f"   Accuracy:  {fused_acc:.4f} ({fused_acc*100:.2f}%)")
    print(f"   Macro F1:  {fused_f1_macro:.4f}")
    print(f"   Weighted F1: {fused_f1_weighted:.4f}")
    results['fused'] = {
        'accuracy': fused_acc,
        'macro_f1': fused_f1_macro,
        'weighted_f1': fused_f1_weighted,
        'predictions': all_fused_preds,
        'probabilities': all_fused_probs,
    }
    
    # Improvement over audio-only
    acc_improvement = fused_acc - audio_acc
    f1_improvement = fused_f1_macro - audio_f1_macro
    
    print(f"\n4. Improvement over Audio-Only:")
    print(f"   Accuracy:  {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
    print(f"   Macro F1:  {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
    
    # Per-class performance
    print(f"\n5. Per-Class Performance (Fused Model):")
    print(classification_report(
        all_true_labels,
        all_fused_preds,
        target_names=config.CANONICAL_EMOTIONS,
        digits=4
    ))
    
    # Confusion matrix
    print(f"\n6. Confusion Matrix (Fused Model):")
    cm = confusion_matrix(all_true_labels, all_fused_preds)
    cm_df = pd.DataFrame(
        cm,
        index=config.CANONICAL_EMOTIONS,
        columns=config.CANONICAL_EMOTIONS
    )
    print(cm_df)
    
    # Per-class accuracy
    print(f"\n7. Per-Class Accuracy (Fused Model):")
    for i, emotion in enumerate(config.CANONICAL_EMOTIONS):
        class_mask = all_true_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_fused_preds[class_mask] == i).mean()
            print(f"   {emotion:12s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_mask.sum()} samples")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fused multimodal emotion model")
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(config.SPLITS_DIR / "test.csv"),
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for faster testing)",
    )
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load models
    models = load_models(device)
    
    # Evaluate
    results = evaluate_models(
        models,
        Path(args.test_csv),
        max_samples=args.max_samples
    )
    
    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()

