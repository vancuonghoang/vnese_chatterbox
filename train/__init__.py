"""
Viterbox Training Package

Provides training utilities for T3 model fine-tuning with LoRA support.
"""

from .loss import T3LossCalculator, LossConfig, compute_t3_loss, create_loss_calculator
from .datasets import SpeechDataCollator, LengthGroupedSampler, CollatorConfig
from .trainer import (
    T3ForFineTuning,
    SafeCheckpointTrainer,
    ResumeVerificationCallback,  # Replaces BestModelCallback
    T3TrainingArguments,
    setup_wandb,
)
from .lora_trainer import (
    LoRAConfig,
    create_lora_model,
    save_lora_weights,
    load_lora_weights,
    merge_lora_weights,
    get_lora_training_args,
)

__all__ = [
    # Loss
    "T3LossCalculator",
    "LossConfig", 
    "compute_t3_loss",
    "create_loss_calculator",
    # Datasets
    "SpeechDataCollator",
    "LengthGroupedSampler",
    "CollatorConfig",
    # Trainer
    "T3ForFineTuning",
    "SafeCheckpointTrainer",
    "ResumeVerificationCallback",  # Replaces BestModelCallback
    "T3TrainingArguments",
    "setup_wandb",
    # LoRA
    "LoRAConfig",
    "create_lora_model",
    "save_lora_weights",
    "load_lora_weights",
    "merge_lora_weights",
    "get_lora_training_args",
]

