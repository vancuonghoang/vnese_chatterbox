"""
Script train đơn giản cho Vietnamese TTS
Chỉ cần: python train.py --csv metadata.csv --audio_dir ./

Features:
- SOTA Loss Functions (Label Smoothing, Loss Weighting, Focal Loss, ZLoss)
- Gradient Checkpointing (~30% VRAM savings)
- Dynamic Batching (group by length, reduce padding waste)
- Best Model Checkpoint (save based on eval_loss_speech)
- On-the-fly Caching (4-5x speedup after epoch 1)
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import multiprocessing

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be done before any other imports that might use multiprocessing
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import HfArgumentParser
from src.finetune_t3_thai import (
    ModelArguments,
    DataArguments,
    CustomTrainingArguments,
    run_training
)


def main():
    parser = argparse.ArgumentParser(description="Train Vietnamese TTS")

    # Data arguments - support both single CSV and separate train/val CSVs
    parser.add_argument("--csv", type=str, help="Path to metadata CSV file (will be split for train/val)")
    parser.add_argument("--train_csv", type=str, help="Path to train metadata CSV file")
    parser.add_argument("--val_csv", type=str, help="Path to validation metadata CSV file")
    parser.add_argument("--audio_dir", type=str, default=".", help="Directory containing audio files (default: same as CSV)")

    # Optional training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vietnamese", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps (default: 5000)")
    parser.add_argument("--eval_steps", type=int, default=5000, help="Evaluate every N steps (default: 5000)")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps (default: -1 for full training)")
    
    # Caching arguments
    parser.add_argument("--use_cache", action="store_true", help="Enable on-the-fly caching (epoch 1 slow, epoch 2+ fast)")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to store cached embeddings (default: ./cache)")
    parser.add_argument("--cache_device", type=str, default="cuda", help="Device for computing cached embeddings: cuda or cpu (default: cuda)")
    
    # Mixed precision
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision training")
    
    # === NEW SOTA FEATURES ===
    
    # Gradient Checkpointing (saves ~30% VRAM)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to save ~30%% VRAM (default: enabled)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing")
    
    # Dynamic Batching (group samples by length)
    parser.add_argument("--use_dynamic_batching", action="store_true", default=True,
                        help="Enable dynamic batching to group samples by similar length (default: enabled)")
    parser.add_argument("--no_dynamic_batching", action="store_true",
                        help="Disable dynamic batching")
    parser.add_argument("--bucket_size_multiplier", type=int, default=100,
                        help="Bucket size multiplier for dynamic batching (default: 100)")
    
    # Best Model Checkpoint
    parser.add_argument("--save_best_model", action="store_true", default=True,
                        help="Save best model based on evaluation metric (default: enabled)")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss_speech",
                        choices=["eval_loss", "eval_loss_speech", "eval_loss_text", "eval_speech_accuracy"],
                        help="Metric to use for best model selection (default: eval_loss_speech)")
    parser.add_argument("--greater_is_better", action="store_true", default=False,
                        help="Whether higher metric value is better (default: False for loss)")
    
    # SOTA Loss Functions
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default: 0.1, 0.0 to disable)")
    parser.add_argument("--text_loss_weight", type=float, default=0.1,
                        help="Weight for text loss (default: 0.1)")
    parser.add_argument("--speech_loss_weight", type=float, default=1.0,
                        help="Weight for speech loss (default: 1.0)")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use Focal Loss for handling rare tokens (Vietnamese tones)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for Focal Loss (default: 2.0)")
    parser.add_argument("--use_zloss", action="store_true",
                        help="Use Z-Loss for numerical stability (from PaLM)")
    parser.add_argument("--zloss_weight", type=float, default=1e-4,
                        help="Weight for Z-Loss (default: 1e-4)")
    
    # Early Stopping
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Early stopping patience (number of evals without improvement)")

    args = parser.parse_args()
    
    # Handle negation flags
    if args.no_gradient_checkpointing:
        args.gradient_checkpointing = False
    if args.no_dynamic_batching:
        args.use_dynamic_batching = False

    # Validate inputs - check if using separate train/val or single CSV
    use_separate_files = args.train_csv and args.val_csv
    use_single_file = args.csv

    if not use_separate_files and not use_single_file:
        print("❌ Error: You must provide either:")
        print("   1. --csv for single file (will be split), OR")
        print("   2. Both --train_csv and --val_csv for separate files")
        return

    if use_separate_files and use_single_file:
        print("⚠️  Warning: Both --csv and --train_csv/--val_csv provided. Using separate files.")

    # Validate file existence
    if use_separate_files:
        train_csv_path = Path(args.train_csv)
        val_csv_path = Path(args.val_csv)

        if not train_csv_path.exists():
            print(f"❌ Train CSV file not found: {args.train_csv}")
            return

        if not val_csv_path.exists():
            print(f"❌ Validation CSV file not found: {args.val_csv}")
            return
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"❌ CSV file not found: {args.csv}")
            return

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {args.audio_dir}")
        return
    
    # Check Vietnamese tokenizer
    tokenizer_path = Path("VietnameseTokenizer/tokenizer.json")
    if not tokenizer_path.exists():
        print("❌ Vietnamese tokenizer not found!")
        print("   Creating Vietnamese tokenizer...")
        os.system("python create_vietnamese_tokenizer.py")
        if not tokenizer_path.exists():
            print("❌ Failed to create tokenizer!")
            return
    
    print("="*80)
    print("VIETNAMESE TTS TRAINING")
    print("="*80)

    if use_separate_files:
        print(f"\n📁 Train CSV: {train_csv_path}")
        print(f"📁 Val CSV: {val_csv_path}")
    else:
        print(f"\n📁 CSV file: {csv_path}")

    print(f"📁 Audio directory: {audio_dir}")
    print(f"🔤 Tokenizer: {tokenizer_path}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.lr}")
    print(f"🔄 Epochs: {args.epochs}")
    if args.max_steps > 0:
        print(f"⚡ Max steps: {args.max_steps} (will override epochs)")
    print(f"💾 Save every: {args.save_steps} steps")
    print(f"📊 Eval every: {args.eval_steps} steps")
    
    # Print caching info
    if args.use_cache:
        print(f"\n📦 Caching: ENABLED")
        print(f"   Cache dir: {args.cache_dir}")
        print(f"   Cache device: {args.cache_device}")
        print(f"   ⚡ Epoch 1: Slow (building cache)")
        print(f"   ⚡ Epoch 2+: Fast (4-5x speedup!)")
    else:
        print(f"\n📦 Caching: DISABLED (consider using --use_cache for 4-5x speedup)")
    
    if args.fp16:
        print(f"🔢 Mixed precision: FP16")
    else:
        print(f"🔢 Mixed precision: BF16")
    
    # Print SOTA features
    print(f"\n{'─'*40}")
    print("🚀 SOTA FEATURES")
    print(f"{'─'*40}")
    
    # Gradient Checkpointing
    if args.gradient_checkpointing:
        print(f"✅ Gradient Checkpointing: ENABLED (~30% VRAM savings)")
    else:
        print(f"❌ Gradient Checkpointing: DISABLED")
    
    # Dynamic Batching
    if args.use_dynamic_batching:
        print(f"✅ Dynamic Batching: ENABLED (bucket_size_multiplier={args.bucket_size_multiplier})")
    else:
        print(f"❌ Dynamic Batching: DISABLED")
    
    # Best Model Checkpoint
    if args.save_best_model:
        print(f"✅ Best Model Checkpoint: ENABLED")
        print(f"   Metric: {args.metric_for_best_model}")
        print(f"   Greater is better: {args.greater_is_better}")
    else:
        print(f"❌ Best Model Checkpoint: DISABLED")
    
    # Loss Functions
    print(f"\n📉 Loss Configuration:")
    print(f"   Label Smoothing: {args.label_smoothing}")
    print(f"   Text Loss Weight: {args.text_loss_weight}")
    print(f"   Speech Loss Weight: {args.speech_loss_weight}")
    if args.use_focal_loss:
        print(f"   ✅ Focal Loss: ENABLED (gamma={args.focal_gamma})")
    if args.use_zloss:
        print(f"   ✅ Z-Loss: ENABLED (weight={args.zloss_weight})")
    
    # Early Stopping
    if args.early_stopping_patience:
        print(f"\n⏱️  Early Stopping: patience={args.early_stopping_patience}")
    
    print("="*80 + "\n")

    # Count samples
    if use_separate_files:
        with open(train_csv_path, 'r', encoding='utf-8') as f:
            train_lines = f.readlines()
            num_train_samples = len(train_lines) - 1  # Exclude header

        with open(val_csv_path, 'r', encoding='utf-8') as f:
            val_lines = f.readlines()
            num_val_samples = len(val_lines) - 1  # Exclude header

        print(f"📊 Found {num_train_samples} training samples")
        print(f"📊 Found {num_val_samples} validation samples")

        # Check a few audio files from train set
        print("\n🔍 Checking training audio files...")
        missing_count = 0
        for i, line in enumerate(train_lines[1:6]):  # Check first 5
            parts = line.strip().split('|')
            if len(parts) >= 2:
                audio_file = parts[0]
                audio_path = audio_dir / audio_file
                if audio_path.exists():
                    print(f"  ✓ {audio_file}")
                else:
                    print(f"  ✗ {audio_file} - NOT FOUND")
                    missing_count += 1
    else:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            num_samples = len(lines) - 1  # Exclude header

        print(f"📊 Found {num_samples} samples in CSV")

        # Check a few audio files
        print("\n🔍 Checking audio files...")
        missing_count = 0
        for i, line in enumerate(lines[1:6]):  # Check first 5
            parts = line.strip().split('|')
            if len(parts) >= 2:
                audio_file = parts[0]
                audio_path = audio_dir / audio_file
                if audio_path.exists():
                    print(f"  ✓ {audio_file}")
                else:
                    print(f"  ✗ {audio_file} - NOT FOUND")
                    missing_count += 1

    if missing_count > 0:
        response = input(f"\n⚠️  Some audio files not found. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    print("\n🚀 Starting training...\n")
    
    # Create model arguments with SOTA loss config
    model_args = ModelArguments(
        model_name_or_path="ResembleAI/chatterbox",
        cache_dir="./cache",
        freeze_voice_encoder=True,
        freeze_s3gen=True,
        tokenizer_path=str(tokenizer_path),
        gradient_checkpointing=args.gradient_checkpointing,
        # SOTA Loss configuration
        label_smoothing=args.label_smoothing,
        text_loss_weight=args.text_loss_weight,
        speech_loss_weight=args.speech_loss_weight,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_z_loss=args.use_zloss,
        z_loss_weight=args.zloss_weight,
    )
    
    # Create data arguments
    if use_separate_files:
        data_args = DataArguments(
            train_metadata_file=str(train_csv_path),
            val_metadata_file=str(val_csv_path),
            audio_dir=str(audio_dir),
            dataset_dir=None,
            dataset_name=None,
            eval_split_size=0.0,  # Not used when separate files provided
            max_text_len=256,
            max_speech_len=1200,
            audio_prompt_duration_s=3.0,
            preprocessing_num_workers=8,
            ignore_verifications=True,
            use_streaming=False,
            # Caching arguments
            use_cache=args.use_cache,
            cache_dir=args.cache_dir,
            cache_device=args.cache_device,
            # Dynamic batching
            use_dynamic_batching=args.use_dynamic_batching,
            bucket_size_multiplier=args.bucket_size_multiplier,
        )
    else:
        data_args = DataArguments(
            metadata_file=str(csv_path),
            audio_dir=str(audio_dir),
            dataset_dir=None,
            dataset_name=None,
            eval_split_size=0,
            max_text_len=256,
            max_speech_len=1200,
            audio_prompt_duration_s=3.0,
            preprocessing_num_workers=8,
            ignore_verifications=True,
            use_streaming=False,
            # Caching arguments
            use_cache=args.use_cache,
            cache_dir=args.cache_dir,
            cache_device=args.cache_device,
            # Dynamic batching
            use_dynamic_batching=args.use_dynamic_batching,
            bucket_size_multiplier=args.bucket_size_multiplier,
        )
    
    # Adjust num_workers for CUDA caching compatibility
    num_workers = 8
    if args.use_cache and args.cache_device == 'cuda':
        print("\n⚠️  CUDA caching enabled: Setting dataloader_num_workers=0")
        print("   (CUDA cannot be used in forked subprocesses)")
        print("   Cache will be computed on main process with GPU")
        num_workers = 0

    # Create training arguments
    training_args = CustomTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,

        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.lr,
        warmup_steps=5000,
        lr_scheduler_type="cosine",

        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,

        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        logging_first_step=True,
        do_train=True,
        do_eval=True if (use_separate_files or args.csv) else False,
        eval_strategy="steps" if (use_separate_files or args.csv) else "no",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        data_seed=42,
        bf16=True if not args.fp16 else False,
        fp16=args.fp16,
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=True if num_workers > 0 else False,
        seed=42,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        
        # Best Model Checkpoint settings
        load_best_model_at_end=args.save_best_model,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        
        # Early Stopping
        early_stopping_patience=args.early_stopping_patience,
    )
    
    # Run training
    try:
        run_training(model_args, data_args, training_args)
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED!")
        print("="*80)
        print(f"\n📁 Model saved at: {args.output_dir}")
        if args.save_best_model:
            print(f"🏆 Best model at: {args.output_dir}/best_model")
        print(f"📊 Logs at: {args.output_dir}/logs")
        print("\n💡 To test the model:")
        print(f"   python test.py --model {args.output_dir} --text 'Xin chào'")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

