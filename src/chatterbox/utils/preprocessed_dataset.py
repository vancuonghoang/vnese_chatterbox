"""
Fast dataset loader for pre-computed embeddings and tokens

This dataset loads from .pt files created by preprocess_dataset.py
Much faster than computing embeddings on-the-fly during training.

Features:
- Parallel loading with multiprocessing workers
- Progress bars with tqdm
- Memory-efficient lazy loading or preload-to-memory option
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple
import logging
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import os

logger = logging.getLogger(__name__)


def _load_single_file(args: Tuple[Path, Optional[int], Optional[int]]) -> Optional[Dict]:
    """
    Worker function to load a single .pt file
    Used by multiprocessing pool
    """
    file_path, max_text_len, max_speech_len = args
    try:
        data = torch.load(file_path, weights_only=False, map_location='cpu')
        
        # Apply length filters
        text_len = data['text_token_lens'].item() if hasattr(data['text_token_lens'], 'item') else data['text_token_lens']
        speech_len = data['speech_token_lens'].item() if hasattr(data['speech_token_lens'], 'item') else data['speech_token_lens']
        
        if max_text_len and text_len > max_text_len:
            return None
        
        if max_speech_len and speech_len > max_speech_len:
            return None
        
        return data
    except Exception as e:
        return None


class PrecomputedDataset(Dataset):
    """
    Fast dataset that loads pre-computed embeddings and tokens from .pt files
    
    Args:
        preprocessed_dir: Directory containing .pt files from preprocess_dataset.py
        max_text_len: Maximum text token length (optional, for additional filtering)
        max_speech_len: Maximum speech token length (optional, for additional filtering)
        preload_to_memory: If True, load all data to RAM at init (faster training, uses more RAM)
        show_progress: If True, show progress bar during loading
        num_workers: Number of parallel workers for loading (default: auto-detect)
        use_threading: Use threading instead of multiprocessing (better for I/O bound, NFS)
    """
    
    def __init__(
        self,
        preprocessed_dir: str,
        max_text_len: Optional[int] = None,
        max_speech_len: Optional[int] = None,
        preload_to_memory: bool = False,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
        use_threading: bool = False,
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.max_text_len = max_text_len
        self.max_speech_len = max_speech_len
        self.preload_to_memory = preload_to_memory
        self.show_progress = show_progress
        self.use_threading = use_threading
        
        # Auto-detect number of workers
        if num_workers is None:
            self.num_workers = min(mp.cpu_count(), 8)  # Cap at 8 for I/O bound tasks
        else:
            self.num_workers = num_workers
        
        # Cached data (if preload_to_memory=True)
        self._cached_data: Optional[List[Dict]] = None
        
        start_time = time.time()
        logger.info(f"🔍 Scanning preprocessed directory: {preprocessed_dir}")
        
        # Find all .pt files
        all_files = list(self.preprocessed_dir.glob("*.pt"))
        
        # Filter out summary file and sort
        self.data_files = sorted([f for f in all_files if 'summary' not in f.name])
        
        if len(self.data_files) == 0:
            raise ValueError(f"❌ No .pt files found in {preprocessed_dir}")
        
        scan_time = time.time() - start_time
        logger.info(f"📁 Found {len(self.data_files):,} preprocessed files ({scan_time:.1f}s)")
        
        # Load and validate first sample
        logger.info("🔬 Validating data format...")
        try:
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
            
            # Log sample info
            text_len = sample['text_token_lens'].item() if hasattr(sample['text_token_lens'], 'item') else sample['text_token_lens']
            speech_len = sample['speech_token_lens'].item() if hasattr(sample['speech_token_lens'], 'item') else sample['speech_token_lens']
            logger.info(f"   Sample 0: text_len={text_len}, speech_len={speech_len}")
            logger.info(f"✅ Data format validation passed")
        except Exception as e:
            raise ValueError(f"❌ Invalid preprocessed data format: {e}")
        
        # Preload all data to memory if requested
        if self.preload_to_memory:
            self._preload_all_data()
        
        total_time = time.time() - start_time
        logger.info(f"🎉 PrecomputedDataset ready! ({len(self):,} samples, {total_time:.1f}s)")
    
    def _preload_all_data(self):
        """Load all .pt files into memory for faster access during training"""
        logger.info(f"📥 Preloading {len(self.data_files):,} files to memory...")
        logger.info(f"   Using {self.num_workers} {'threads' if self.use_threading else 'processes'}")
        
        self._cached_data = []
        skipped_count = 0
        error_count = 0
        
        start_time = time.time()
        
        if self.num_workers > 1:
            # Parallel loading with workers
            self._cached_data = self._parallel_load()
            skipped_count = len(self.data_files) - len(self._cached_data)
        else:
            # Sequential loading (original behavior)
            self._cached_data = self._sequential_load()
            skipped_count = len(self.data_files) - len(self._cached_data)
        
        load_time = time.time() - start_time
        
        # Update data_files to match cached data
        self.data_files = self.data_files[:len(self._cached_data)]
        
        logger.info(f"📦 Preloaded {len(self._cached_data):,} samples to memory ({load_time:.1f}s)")
        if skipped_count > 0:
            logger.info(f"   Skipped {skipped_count:,} samples (exceeded length limits or errors)")
        
        # Estimate memory usage
        if len(self._cached_data) > 0:
            sample_size = sum(
                v.element_size() * v.numel() if torch.is_tensor(v) else 0
                for v in self._cached_data[0].values()
            )
            total_mb = (sample_size * len(self._cached_data)) / (1024 * 1024)
            logger.info(f"   Estimated memory usage: {total_mb:.1f} MB")
            
        # Show speed stats
        files_per_sec = len(self.data_files) / load_time if load_time > 0 else 0
        logger.info(f"   Loading speed: {files_per_sec:.1f} files/sec")
    
    def _parallel_load(self) -> List[Dict]:
        """Load files in parallel using multiprocessing or threading"""
        loaded_data = []
        
        # Prepare arguments for workers
        load_args = [
            (f, self.max_text_len, self.max_speech_len) 
            for f in self.data_files
        ]
        
        # Choose executor based on use_threading flag
        ExecutorClass = ThreadPoolExecutor if self.use_threading else ProcessPoolExecutor
        
        with ExecutorClass(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_load_single_file, args): i 
                for i, args in enumerate(load_args)
            }
            
            # Process results with progress bar
            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Loading ({self.num_workers} workers)",
                unit="files",
                disable=not self.show_progress,
                dynamic_ncols=True,
            )
            
            # Collect results maintaining order
            results = [None] * len(self.data_files)
            loaded_count = 0
            skipped_count = 0
            
            for future in pbar:
                idx = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[idx] = result
                        loaded_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    skipped_count += 1
                
                pbar.set_postfix({
                    'loaded': loaded_count,
                    'skipped': skipped_count,
                })
            
            pbar.close()
        
        # Filter out None values while maintaining order
        loaded_data = [r for r in results if r is not None]
        
        return loaded_data
    
    def _sequential_load(self) -> List[Dict]:
        """Load files sequentially (fallback when workers=1)"""
        loaded_data = []
        skipped_count = 0
        error_count = 0
        
        pbar = tqdm(
            self.data_files,
            desc="Loading data",
            unit="files",
            disable=not self.show_progress,
            dynamic_ncols=True,
        )
        
        for data_file in pbar:
            try:
                data = torch.load(data_file, weights_only=False)
                
                # Apply length filters
                text_len = data['text_token_lens'].item() if hasattr(data['text_token_lens'], 'item') else data['text_token_lens']
                speech_len = data['speech_token_lens'].item() if hasattr(data['speech_token_lens'], 'item') else data['speech_token_lens']
                
                if self.max_text_len and text_len > self.max_text_len:
                    skipped_count += 1
                    continue
                
                if self.max_speech_len and speech_len > self.max_speech_len:
                    skipped_count += 1
                    continue
                
                loaded_data.append(data)
                
                pbar.set_postfix({
                    'loaded': len(loaded_data),
                    'skipped': skipped_count,
                    'errors': error_count,
                })
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    logger.warning(f"⚠️ Error loading {data_file.name}: {e}")
        
        pbar.close()
        return loaded_data
    
    def __len__(self):
        if self._cached_data is not None:
            return len(self._cached_data)
        return len(self.data_files)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
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
                
                # Apply length filters (only if not preloaded, as preload already filters)
                text_len = data['text_token_lens'].item() if hasattr(data['text_token_lens'], 'item') else data['text_token_lens']
                speech_len = data['speech_token_lens'].item() if hasattr(data['speech_token_lens'], 'item') else data['speech_token_lens']
                
                if self.max_text_len and text_len > self.max_text_len:
                    return None
                
                if self.max_speech_len and speech_len > self.max_speech_len:
                    return None
            
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
            if self._cached_data is None:
                logger.error(f"Error loading {self.data_files[idx]}: {e}")
            else:
                logger.error(f"Error accessing cached data at index {idx}: {e}")
            return None
    
    def get_metadata(self, idx):
        """Get metadata (text, audio_path) for a sample"""
        try:
            if self._cached_data is not None:
                data = self._cached_data[idx]
            else:
                data = torch.load(self.data_files[idx], weights_only=False)
            return {
                'text': data.get('text', ''),
                'audio_path': data.get('audio_path', ''),
            }
        except:
            return None
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        logger.info("📊 Computing dataset statistics...")
        
        text_lens = []
        speech_lens = []
        
        # Sample up to 1000 items for stats
        sample_size = min(1000, len(self))
        indices = list(range(0, len(self), max(1, len(self) // sample_size)))[:sample_size]
        
        pbar = tqdm(
            indices,
            desc="Computing stats",
            disable=not self.show_progress,
            dynamic_ncols=True,
        )
        
        for idx in pbar:
            try:
                if self._cached_data is not None:
                    data = self._cached_data[idx]
                else:
                    data = torch.load(self.data_files[idx], weights_only=False)
                
                text_len = data['text_token_lens'].item() if hasattr(data['text_token_lens'], 'item') else data['text_token_lens']
                speech_len = data['speech_token_lens'].item() if hasattr(data['speech_token_lens'], 'item') else data['speech_token_lens']
                
                text_lens.append(text_len)
                speech_lens.append(speech_len)
            except:
                pass
        
        pbar.close()
        
        stats = {
            'total_samples': len(self),
            'sampled_for_stats': len(text_lens),
            'text_len': {
                'min': min(text_lens) if text_lens else 0,
                'max': max(text_lens) if text_lens else 0,
                'mean': sum(text_lens) / len(text_lens) if text_lens else 0,
            },
            'speech_len': {
                'min': min(speech_lens) if speech_lens else 0,
                'max': max(speech_lens) if speech_lens else 0,
                'mean': sum(speech_lens) / len(speech_lens) if speech_lens else 0,
            },
        }
        
        logger.info(f"   Total samples: {stats['total_samples']:,}")
        logger.info(f"   Text length: min={stats['text_len']['min']}, max={stats['text_len']['max']}, mean={stats['text_len']['mean']:.1f}")
        logger.info(f"   Speech length: min={stats['speech_len']['min']}, max={stats['speech_len']['max']}, mean={stats['speech_len']['mean']:.1f}")
        
        return stats


def collate_fn(batch):
    """
    Collate function for PrecomputedDataset
    Same as the one in finetune_t3_thai.py
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return {}
    
    # Stack all tensors
    return {
        'text_tokens': torch.nn.utils.rnn.pad_sequence(
            [item['text_tokens'] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        'text_token_lens': torch.stack([item['text_token_lens'] for item in batch]),
        'speech_tokens': torch.nn.utils.rnn.pad_sequence(
            [item['speech_tokens'] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        'speech_token_lens': torch.stack([item['speech_token_lens'] for item in batch]),
        't3_cond_speaker_emb': torch.stack([item['t3_cond_speaker_emb'] for item in batch]),
        't3_cond_prompt_speech_tokens': torch.stack([item['t3_cond_prompt_speech_tokens'] for item in batch]),
        't3_cond_emotion_adv': torch.stack([item['t3_cond_emotion_adv'] for item in batch]),
    }
