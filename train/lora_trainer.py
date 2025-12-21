"""
LoRA Fine-tuning for T3 Model

Implements Parameter-Efficient Fine-Tuning (PEFT) using LoRA adapters.
Optimized for small datasets (2-5h audio).

Usage:
    python -m train.lora_trainer \
        --data_dir ./data \
        --output_dir ./checkpoints \
        --epochs 10 \
        --batch_size 4
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """
    LoRA configuration optimized for 2-5h audio fine-tuning.
    
    Low rank (r=8) works well for small datasets.
    Target attention layers in Llama backbone.
    """
    # LoRA hyperparameters
    r: int = 8  # LoRA rank (8 for small datasets, 16-32 for larger)
    lora_alpha: int = 16  # Scaling factor (typically 2x rank)
    lora_dropout: float = 0.05
    
    # Target modules in Llama backbone
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
    ])
    
    # Training settings for 2-5h audio
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 4  # Fits on T4 16GB
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    warmup_steps: int = 100
    
    # Regularization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Saving
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3


def create_lora_model(base_model, config: Optional[LoRAConfig] = None):
    """
    Apply LoRA adapters to T3 Llama backbone.
    
    Freezes all base model parameters and adds trainable LoRA adapters.
    
    Args:
        base_model: T3 model instance
        config: LoRA configuration
        
    Returns:
        Model with LoRA adapters applied to Llama backbone
    """
    if config is None:
        config = LoRAConfig()
    
    try:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "peft library required for LoRA. Install with: pip install peft"
        )
    
    # Get the transformer backbone from T3
    if hasattr(base_model, "t3"):
        # T3ForFineTuning wrapper
        tfmr = base_model.t3.tfmr
    elif hasattr(base_model, "tfmr"):
        # Direct T3 model
        tfmr = base_model.tfmr
    else:
        raise ValueError("Could not find transformer backbone in model")
    
    # Create PEFT config (no task_type - T3 uses encoder-only LlamaModel)
    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        # task_type removed - not needed for encoder-only models
    )
    
    # Apply LoRA to transformer backbone
    lora_tfmr = get_peft_model(tfmr, peft_config)
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in lora_tfmr.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_tfmr.parameters())
    logger.info(
        f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    
    # Replace backbone
    if hasattr(base_model, "t3"):
        base_model.t3.tfmr = lora_tfmr
    else:
        base_model.tfmr = lora_tfmr
    
    return base_model


def save_lora_weights(model, save_dir: str, save_name: str = "lora_adapter"):
    """
    Save only the LoRA adapter weights (very small file).
    
    Args:
        model: Model with LoRA adapters
        save_dir: Directory to save to
        save_name: Name for the saved adapter
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get the PEFT model
    if hasattr(model, "t3"):
        peft_model = model.t3.tfmr
    elif hasattr(model, "tfmr"):
        peft_model = model.tfmr
    else:
        raise ValueError("Could not find transformer backbone")
    
    # Save adapter
    save_path = os.path.join(save_dir, save_name)
    peft_model.save_pretrained(save_path)
    
    logger.info(f"LoRA adapter saved to {save_path}")
    return save_path


def load_lora_weights(model, adapter_path: str):
    """
    Load LoRA adapter weights into model.
    
    Args:
        model: Base T3 model
        adapter_path: Path to saved adapter
        
    Returns:
        Model with loaded LoRA adapters
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("peft library required. Install with: pip install peft")
    
    # Get the transformer backbone
    if hasattr(model, "t3"):
        tfmr = model.t3.tfmr
    elif hasattr(model, "tfmr"):
        tfmr = model.tfmr
    else:
        raise ValueError("Could not find transformer backbone")
    
    # Load adapter
    peft_model = PeftModel.from_pretrained(tfmr, adapter_path)
    
    # Replace backbone
    if hasattr(model, "t3"):
        model.t3.tfmr = peft_model
    else:
        model.tfmr = peft_model
    
    logger.info(f"LoRA adapter loaded from {adapter_path}")
    return model


def merge_lora_weights(model):
    """
    Merge LoRA weights into base model for faster inference.
    
    After merging, no adapter overhead during inference.
    
    Args:
        model: Model with LoRA adapters
        
    Returns:
        Model with merged weights (no longer requires PEFT)
    """
    # Get the PEFT model
    if hasattr(model, "t3"):
        peft_model = model.t3.tfmr
    elif hasattr(model, "tfmr"):
        peft_model = model.tfmr
    else:
        raise ValueError("Could not find transformer backbone")
    
    # Merge and unload
    merged = peft_model.merge_and_unload()
    
    # Replace backbone
    if hasattr(model, "t3"):
        model.t3.tfmr = merged
    else:
        model.tfmr = merged
    
    logger.info("LoRA weights merged into base model")
    return model


def get_lora_training_args(config: Optional[LoRAConfig] = None, output_dir: str = "./output"):
    """
    Get HuggingFace TrainingArguments configured for LoRA training.
    """
    if config is None:
        config = LoRAConfig()
    
    from trainer import T3TrainingArguments
    
    return T3TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        logging_steps=10,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss_speech",
        greater_is_better=False,
        report_to="wandb",
        use_dynamic_batching=True,
        save_best_model=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point for LoRA training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for T3")
    parser.add_argument("--data_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--model_path", type=str, default=None, help="Base model path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--wandb_project", type=str, default="viterbox-lora", help="WandB project name")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info(f"Starting LoRA training with config: {args}")
    
    # Create config
    lora_config = LoRAConfig(
        r=args.lora_r,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    logger.info(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    logger.info(f"Target modules: {lora_config.target_modules}")
    
    # NOTE: Full training loop implementation would go here
    # This requires the dataset and model loading which depends on viterbox structure
    
    logger.info("LoRA training setup complete. Implement dataset loading to continue.")


if __name__ == "__main__":
    main()
