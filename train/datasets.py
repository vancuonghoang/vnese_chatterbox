"""
Dataset utilities for T3 fine-tuning

Includes:
- SpeechDataCollator: Batching and label masking
- LengthGroupedSampler: Dynamic batching by length
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

IGNORE_ID = -100


@dataclass
class CollatorConfig:
    """Configuration for SpeechDataCollator"""
    speech_cond_prompt_len: int = 150  # Default T3 prompt length
    text_pad_token_id: int = 0
    speech_pad_token_id: int = 0


class SpeechDataCollator:
    """
    Collator for speech fine-tuning datasets.
    
    - Pads text and speech tokens to batch max length
    - Creates labels with IGNORE_ID for prompt and padding
    - Handles CPU/GPU tensor compatibility
    """
    
    def __init__(
        self,
        speech_cond_prompt_len: int = 150,
        text_pad_token_id: int = 0,
        speech_pad_token_id: int = 0,
    ):
        self.speech_cond_prompt_len = speech_cond_prompt_len
        self.text_pad_token_id = text_pad_token_id
        self.speech_pad_token_id = speech_pad_token_id
    
    @classmethod
    def from_config(cls, config: CollatorConfig) -> "SpeechDataCollator":
        return cls(
            speech_cond_prompt_len=config.speech_cond_prompt_len,
            text_pad_token_id=config.text_pad_token_id,
            speech_pad_token_id=config.speech_pad_token_id,
        )
    
    def __call__(self, features: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Collate a batch of features.
        
        Expected feature keys:
        - text_tokens: (seq_len,)
        - speech_tokens: (seq_len,)
        - text_token_lens: scalar or (1,)
        - speech_token_lens: scalar or (1,)
        - t3_cond_speaker_emb: (D_speaker,)
        - t3_cond_prompt_speech_tokens: (prompt_len,)
        - t3_cond_emotion_adv: scalar
        """
        # Filter None features
        valid_features = [f for f in features if f is not None]
        
        if not valid_features:
            logger.warning("SpeechDataCollator received no valid features.")
            return {}
        
        features = valid_features
        batch_size = len(features)
        
        # Ensure all tensors are on CPU (Trainer moves to correct device)
        def to_cpu(t):
            return t.cpu() if torch.is_tensor(t) else torch.tensor(t)
        
        text_tokens_list = [to_cpu(f["text_tokens"]) for f in features]
        speech_tokens_list = [to_cpu(f["speech_tokens"]) for f in features]
        
        max_text_len = max(len(t) for t in text_tokens_list)
        max_speech_len = max(len(t) for t in speech_tokens_list)
        
        # Pad text tokens
        padded_text_tokens = torch.stack([
            F.pad(t, (0, max_text_len - len(t)), value=self.text_pad_token_id)
            for t in text_tokens_list
        ])  # (B, max_text_len)
        
        # Pad speech tokens
        padded_speech_tokens = torch.stack([
            F.pad(s, (0, max_speech_len - len(s)), value=self.speech_pad_token_id)
            for s in speech_tokens_list
        ])  # (B, max_speech_len)
        
        # Collect lengths
        text_token_lens = torch.stack([to_cpu(f["text_token_lens"]) for f in features])
        speech_token_lens = torch.stack([to_cpu(f["speech_token_lens"]) for f in features])
        
        # Collect conditionals
        t3_cond_speaker_emb = torch.stack([to_cpu(f["t3_cond_speaker_emb"]) for f in features])
        t3_cond_prompt_speech_tokens = torch.stack([
            to_cpu(f["t3_cond_prompt_speech_tokens"]) for f in features
        ])
        
        emotion_adv_scalars = torch.stack([to_cpu(f["t3_cond_emotion_adv"]) for f in features])
        t3_cond_emotion_adv = emotion_adv_scalars.view(batch_size, 1, 1)
        
        prompt_len = self.speech_cond_prompt_len
        
        # --- Build labels_text ---
        # Shift off BOS: new length = max_text_len - 1
        shifted_text = padded_text_tokens[:, 1:].contiguous()
        T_text = shifted_text.size(1)
        
        # Mask positions t >= (text_len - 1)
        text_lens_minus_one = (text_token_lens - 1).clamp(min=0)
        arange_text = torch.arange(T_text)
        mask_pad_text = arange_text[None] >= text_lens_minus_one[:, None]
        
        labels_text = shifted_text.clone()
        labels_text[mask_pad_text] = IGNORE_ID
        
        # --- Build labels_speech ---
        # Shift off BOS: new length = max_speech_len - 1
        shifted_speech = padded_speech_tokens[:, 1:].contiguous()
        T_speech = shifted_speech.size(1)
        
        # Mask positions t >= (speech_len - 1)
        speech_lens_minus_one = (speech_token_lens - 1).clamp(min=0)
        arange_speech = torch.arange(T_speech)
        mask_pad_speech = arange_speech[None] >= speech_lens_minus_one[:, None]
        
        # Mask positions t < prompt_len (prompt tokens)
        mask_prompt = arange_speech[None] < prompt_len
        mask_prompt = mask_prompt.expand(batch_size, T_speech)
        
        # Combine masks
        mask_speech_total = mask_pad_speech | mask_prompt
        
        labels_speech = shifted_speech.clone()
        labels_speech[mask_speech_total] = IGNORE_ID
        
        return {
            "text_tokens": padded_text_tokens,
            "text_token_lens": text_token_lens,
            "speech_tokens": padded_speech_tokens,
            "speech_token_lens": speech_token_lens,
            "t3_cond_speaker_emb": t3_cond_speaker_emb,
            "t3_cond_prompt_speech_tokens": t3_cond_prompt_speech_tokens,
            "t3_cond_emotion_adv": t3_cond_emotion_adv,
            "labels_text": labels_text,
            "labels_speech": labels_speech,
        }


class LengthGroupedSampler(Sampler):
    """
    Sampler that groups samples by similar length to minimize padding waste.
    
    Benefits:
    - Reduces memory waste from excessive padding
    - More stable gradients (similar-length samples)
    - Faster training (less wasted compute on padding)
    
    Args:
        dataset: The dataset to sample from
        batch_size: Number of samples per batch
        lengths: Pre-computed lengths for each sample
        shuffle: Whether to shuffle within buckets
        seed: Random seed for shuffling
        bucket_size_multiplier: How many batches to group together
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        lengths: Optional[List[int]] = None,
        shuffle: bool = True,
        seed: int = 42,
        bucket_size_multiplier: int = 100,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = batch_size * bucket_size_multiplier
        
        # Get or compute lengths
        if lengths is not None:
            self.lengths = lengths
        else:
            self.lengths = self._compute_lengths()
        
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
    
    def _compute_lengths(self) -> List[int]:
        """Compute lengths for all samples in dataset."""
        lengths = []
        failed_samples = []
        
        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]
                if sample is None:
                    lengths.append(0)
                    failed_samples.append((i, "Sample is None"))
                elif "speech_tokens" in sample:
                    lengths.append(len(sample["speech_tokens"]))
                else:
                    lengths.append(0)
                    failed_samples.append((i, "Missing 'speech_tokens' key"))
            except (IOError, OSError) as e:
                # File I/O errors
                logger.warning(f"Failed to load sample {i}: I/O error - {e}")
                lengths.append(0)
                failed_samples.append((i, f"I/O error: {e}"))
            except (KeyError, ValueError, RuntimeError) as e:
                # Data format errors
                logger.warning(f"Failed to load sample {i}: Data error - {e}")
                lengths.append(0)
                failed_samples.append((i, f"Data error: {e}"))
            except Exception as e:
                # Catch-all for unexpected errors
                logger.warning(f"Failed to load sample {i}: Unexpected error - {type(e).__name__}: {e}")
                lengths.append(0)
                failed_samples.append((i, f"Unexpected: {type(e).__name__}"))
        
        # Report summary
        if failed_samples:
            logger.warning(f"⚠️  {len(failed_samples)}/{len(self.dataset)} samples failed to load")
            if len(failed_samples) <= 10:
                # Show all if few
                for idx, error in failed_samples:
                    logger.warning(f"  Sample {idx}: {error}")
            else:
                # Show first 5 and last 5
                for idx, error in failed_samples[:5]:
                    logger.warning(f"  Sample {idx}: {error}")
                logger.warning(f"  ... ({len(failed_samples) - 10} more)")
                for idx, error in failed_samples[-5:]:
                    logger.warning(f"  Sample {idx}: {error}")
        
        return lengths
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        # Sort by length
        indices_with_lengths = [(i, self.lengths[i]) for i in indices]
        indices_with_lengths.sort(key=lambda x: x[1])
        
        # Create buckets
        buckets = []
        for i in range(0, len(indices_with_lengths), self.bucket_size):
            bucket = [idx for idx, _ in indices_with_lengths[i:i + self.bucket_size]]
            buckets.append(bucket)
        
        # Shuffle within each bucket
        if self.shuffle:
            for bucket in buckets:
                for i in range(len(bucket) - 1, 0, -1):
                    j = torch.randint(0, i + 1, (1,), generator=self.generator).item()
                    bucket[i], bucket[j] = bucket[j], bucket[i]
        
        # Shuffle bucket order
        if self.shuffle:
            bucket_order = list(range(len(buckets)))
            for i in range(len(bucket_order) - 1, 0, -1):
                j = torch.randint(0, i + 1, (1,), generator=self.generator).item()
                bucket_order[i], bucket_order[j] = bucket_order[j], bucket_order[i]
            buckets = [buckets[i] for i in bucket_order]
        
        # Flatten and yield
        for bucket in buckets:
            for idx in bucket:
                yield idx
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def set_epoch(self, epoch: int):
        """Update seed for each epoch to ensure different shuffling."""
        self.generator.manual_seed(self.seed + epoch)


# =============================================================================
# PrecomputedDataset for loading .pt files
# =============================================================================

from pathlib import Path
from torch.utils.data import Dataset

class PrecomputedDataset(Dataset):
    """
    Fast dataset that loads pre-computed embeddings and tokens from .pt files
    
    Args:
        preprocessed_dir: Directory containing .pt files from preprocess_dataset.py
        preload_to_memory: If True, load all data to RAM at init (faster training)
    """
    
    def __init__(
        self,
        preprocessed_dir: str,
        preload_to_memory: bool = False,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.preload_to_memory = preload_to_memory
        self._cached_data = None
        
        logger.info(f"🔍 Scanning preprocessed directory: {preprocessed_dir}")
        
        # Find all .pt files
        all_files = list(self.preprocessed_dir.glob("*.pt"))
        
        # Filter out summary file and sort
        self.data_files = sorted([f for f in all_files if 'summary' not in f.name])
        
        if len(self.data_files) == 0:
            raise ValueError(f"❌ No .pt files found in {preprocessed_dir}")
        
        logger.info(f"📁 Found {len(self.data_files):,} preprocessed files")
        
        # Load and validate first sample
        logger.info("🔬 Validating data format...")
        sample = torch.load(self.data_files[0], weights_only=False)
        required_keys = [
            'text_tokens', 'text_token_lens',
            'speech_tokens', 'speech_token_lens',
            't3_cond_speaker_emb', 't3_cond_prompt_speech_tokens',
            't3_cond_emotion_adv'
        ]
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        logger.info(f"✅ Data format validation passed")
        
        # Preload all data to memory if requested
        if self.preload_to_memory:
            logger.info(f"📥 Preloading {len(self.data_files):,} files to memory...")
            self._cached_data = []
            for data_file in self.data_files:
                try:
                    data = torch.load(data_file, weights_only=False)
                    self._cached_data.append(data)
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {data_file.name}: {e}")
            logger.info(f"📦 Preloaded {len(self._cached_data):,} samples to memory")
        
        logger.info(f"🎉 PrecomputedDataset ready! ({len(self):,} samples)")
    
    def __len__(self):
        if self._cached_data is not None:
            return len(self._cached_data)
        return len(self.data_files)
    
    def __getitem__(self, idx):
        """
        Load pre-computed data from .pt file or cache
        
        Returns:
            Dictionary with all required fields for training
        """
        try:
            # Use cached data if available
            if self._cached_data is not None:
                data = self._cached_data[idx]
            else:
                data_file = self.data_files[idx]
                data = torch.load(data_file, weights_only=False)
            
            # Return in the format expected by training
            return {
                'text_tokens': data['text_tokens'],
                'text_token_lens': data['text_token_lens'],
                'speech_tokens': data['speech_tokens'],
                'speech_token_lens': data['speech_token_lens'],
                't3_cond_speaker_emb': data['t3_cond_speaker_emb'],
                't3_cond_prompt_speech_tokens': data['t3_cond_prompt_speech_tokens'],
                't3_cond_emotion_adv': data['t3_cond_emotion_adv'],
            }
            
        except Exception as e:
            logger.error(f"Error loading {self.data_files[idx]}: {e}")
            return None
