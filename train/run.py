"""
Unified Training Script for Vietnamese TTS (Pre-computed Flow Only)
Located in viterbox-tts/train/run.py

This script uses ONLY the Pre-computed approach for production training:
1. Run preprocess_dataset.py first to create .pt files
2. Run this script with --preprocessed_dir

Usage:
    python viterbox-tts/train/run.py \
        --preprocessed_dir ./preprocessed \
        --output_dir ./checkpoints/vietnamese \
        --use_wandb

Author: Cuong Hoang
Date: 2025-12-10
"""

import os
import sys
import json
import logging
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import torch
from transformers import (
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Add parent directory to path (where viterbox package is)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Add local train package to path (for direct imports)
TRAIN_PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_PKG))

try:
    from viterbox.tts import Viterbox, REPO_ID
except ImportError as e:
    print(f"❌ Critical Error: Could not import 'viterbox': {e}")
    print("Make sure you have installed the package: pip install -e .")
    sys.exit(1)

# Import from local package
from loss import T3LossCalculator
from datasets import SpeechDataCollator, LengthGroupedSampler, PrecomputedDataset
from trainer import (
    T3ForFineTuning, 
    SafeCheckpointTrainer, 
    ResumeVerificationCallback,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Vietnamese TTS with Pre-computed Data")
    
    # Data
    parser.add_argument("--preprocessed_dir", type=str, required=True,
                        help="Directory with preprocessed .pt files from preprocess_dataset.py")
    parser.add_argument("--val_preprocessed_dir", type=str, default=None,
                        help="Optional: separate validation preprocessed directory")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio if no separate val dir")
    
    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Viterbox checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, default="VietnameseTokenizer/tokenizer.json",
                        help="Vietnamese tokenizer path")
    
    # Training
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vietnamese")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Loss
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--text_weight", type=float, default=0.1)
    parser.add_argument("--speech_weight", type=float, default=1.0)
    parser.add_argument("--use_zloss", action="store_true", default=True)
    parser.add_argument("--zloss_weight", type=float, default=1e-4)
    
    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--dynamic_batching", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="vietnamese-tts")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # LoRA (optional)
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    
    # Validate
    if not Path(args.preprocessed_dir).exists():
        logger.error(f"❌ Preprocessed dir not found: {args.preprocessed_dir}")
        return
    
    print("=" * 80)
    print("VIETNAMESE TTS TRAINING (Pre-computed Flow)")
    print("=" * 80)
    print(f"📁 Preprocessed data: {args.preprocessed_dir}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"LoRA: {args.use_lora}")
    print("=" * 80)
    
    # Initialize WandB
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args),
            )
            logger.info("✅ WandB initialized")
        except Exception as e:
            logger.warning(f"WandB init failed: {e}")
            args.use_wandb = False
    
    # Load dataset
    logger.info("Loading preprocessed dataset...")
    train_dataset = PrecomputedDataset(
        preprocessed_dir=args.preprocessed_dir,
        preload_to_memory=False,
    )
    logger.info(f"✅ Loaded {len(train_dataset)} samples")
    
    # Split for validation
    eval_dataset = None
    if args.val_preprocessed_dir:
        eval_dataset = PrecomputedDataset(
            preprocessed_dir=args.val_preprocessed_dir,
            preload_to_memory=False,
        )
        logger.info(f"✅ Loaded {len(eval_dataset)} validation samples")
    elif args.val_split > 0:
        total = len(train_dataset)
        val_size = int(total * args.val_split)
        train_size = total - val_size
        
        indices = list(range(total))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        from torch.utils.data import Subset
        eval_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)
        logger.info(f"✅ Split: {train_size} train, {val_size} val")
    
    # Load model
    logger.info("Loading Viterbox model...")
    checkpoint_dir = args.checkpoint
    if not checkpoint_dir:
        # Download from HuggingFace
        from huggingface_hub import snapshot_download
        checkpoint_dir = Path(args.output_dir) / "pretrained_model_download"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            REPO_ID,
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False,
        )
    
    # Copy tokenizer if needed
    if Path(args.tokenizer_path).exists():
        import shutil
        target = Path(checkpoint_dir) / "tokenizer.json"
        if not target.exists():
            shutil.copy(args.tokenizer_path, target)
    
    viterbox = Viterbox.from_local(str(checkpoint_dir), device="cpu")
    t3_model = viterbox.t3
    t3_config = t3_model.hp
    
    # Freeze encoder/decoder
    for p in viterbox.ve.parameters():
        p.requires_grad = False
    for p in viterbox.s3gen.parameters():
        p.requires_grad = False
    for p in t3_model.parameters():
        p.requires_grad = True
    
    logger.info("✅ Voice Encoder frozen")
    logger.info("✅ S3Gen frozen")
    logger.info("✅ T3 trainable")
    
    # Gradient checkpointing
    if args.gradient_checkpointing:
        if hasattr(t3_model.tfmr, 'gradient_checkpointing_enable'):
            t3_model.tfmr.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
    
    # LoRA (optional)
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model
            # Don't use task_type - T3 uses LlamaModel (encoder only), not LlamaForCausalLM
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                # task_type removed - not needed for encoder-only T3 model
            )
            # Apply LoRA to the transformer backbone (tfmr, not llama)
            t3_model.tfmr = get_peft_model(t3_model.tfmr, lora_config)
            trainable = sum(p.numel() for p in t3_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in t3_model.parameters())
            logger.info(f"✅ LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
        except ImportError:
            logger.warning("⚠️ peft not installed, skipping LoRA")
            args.use_lora = False
    
    # Wrap model
    model = T3ForFineTuning(
        t3_model=t3_model,
        t3_config=t3_config,
        label_smoothing=args.label_smoothing,
        text_weight=args.text_weight,
        speech_weight=args.speech_weight,
        use_z_loss=args.use_zloss,
        z_loss_weight=args.zloss_weight,
    )
    
    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=args.max_grad_norm,
        logging_steps=50,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        dataloader_num_workers=4,
        report_to=["wandb"] if args.use_wandb else ["tensorboard"],
        remove_unused_columns=False,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        seed=42,
    )
    
    # Collator
    collator = SpeechDataCollator(speech_cond_prompt_len=t3_config.speech_cond_prompt_len)
    
    # Callbacks
    callbacks = [
        ResumeVerificationCallback(),  # Logs resume details
    ]
    
    # Trainer
    trainer = SafeCheckpointTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        use_dynamic_batching=args.dynamic_batching,
    )
    
    # Resume
    last_ckpt = get_last_checkpoint(args.output_dir)
    if last_ckpt:
        logger.info(f"Resuming from {last_ckpt}")
    
    # Train
    logger.info("🚀 Starting training...")
    set_seed(42)
    
    try:
        trainer.train(resume_from_checkpoint=last_ckpt)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED!")
        print("=" * 80)
        print(f"📁 Model: {args.output_dir}")
        print(f"🏆 Best model: {args.output_dir}/best_model")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
