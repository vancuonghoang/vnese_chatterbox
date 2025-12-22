"""
T3 Trainer with WandB logging and Best Model Saving

Features:
- T3ForFineTuning wrapper for HuggingFace Trainer
- SafeCheckpointTrainer with dynamic batching
- BestModelCallback for auto-saving best checkpoints
- WandB integration for experiment tracking
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Trainer, 
    TrainingArguments, 
    TrainerCallback,
    PretrainedConfig,
)
from transformers.trainer_callback import TrainerState, TrainerControl

from loss import T3LossCalculator, LossConfig
from datasets import LengthGroupedSampler

logger = logging.getLogger(__name__)

IGNORE_ID = -100


@dataclass
class T3TrainingArguments(TrainingArguments):
    """Extended training arguments with custom options"""
    # Dynamic batching
    use_dynamic_batching: bool = field(
        default=True,
        metadata={"help": "Use LengthGroupedSampler for dynamic batching"}
    )
    bucket_size_multiplier: int = field(
        default=100,
        metadata={"help": "Multiplier for bucket size in dynamic batching"}
    )
    
    # Best model saving
    save_best_model: bool = field(
        default=True,
        metadata={"help": "Save best model based on eval metric"}
    )
    metric_for_best_model: str = field(
        default="eval_loss_speech",
        metadata={"help": "Metric to use for selecting best model"}
    )
    
    # WandB
    report_to: str = field(
        default="wandb",
        metadata={"help": "Reporting integration (wandb, tensorboard, none)"}
    )


class T3ForFineTuning(nn.Module):
    """
    Wrapper for T3 model with SOTA loss functions for fine-tuning.
    
    Compatible with HuggingFace Trainer.
    """
    
    def __init__(
        self,
        t3_model,
        t3_config,
        # Loss configuration
        label_smoothing: float = 0.1,
        text_weight: float = 0.1,
        speech_weight: float = 1.0,
        use_z_loss: bool = True,
        z_loss_weight: float = 1e-4,
    ):
        super().__init__()
        self.t3 = t3_model
        self.t3_config = t3_config
        
        # For logging
        self.last_loss_text = None
        self.last_loss_speech = None
        
        # Initialize loss calculator
        self.loss_calculator = T3LossCalculator(
            text_weight=text_weight,
            speech_weight=speech_weight,
            label_smoothing=label_smoothing,
            use_z_loss=use_z_loss,
            z_loss_weight=z_loss_weight,
        )
        
        logger.info(f"T3ForFineTuning initialized with SOTA loss")
        
        # Create HF-compatible config for serialization
        class HFCompatibleConfig(PretrainedConfig):
            model_type = "viterbox_t3_finetune"
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
        
        self.config = HFCompatibleConfig()
    
    @property
    def device(self):
        return self.t3.device
    
    def forward(
        self,
        text_tokens,
        text_token_lens,
        speech_tokens,
        speech_token_lens,
        t3_cond_speaker_emb,
        t3_cond_prompt_speech_tokens,
        t3_cond_emotion_adv,
        labels_text=None,
        labels_speech=None,
    ):
        """
        Forward pass with loss computation.
        
        Returns:
            total_loss: Scalar loss for optimization
            speech_logits: For metrics computation
        """
        # Build T3Cond
        # Note: Import here to avoid circular dependency
        from viterbox.models.t3.modules.cond_enc import T3Cond
        
        t3_cond = T3Cond(
            speaker_emb=t3_cond_speaker_emb,
            cond_prompt_speech_tokens=t3_cond_prompt_speech_tokens,
            cond_prompt_speech_emb=None,
            emotion_adv=t3_cond_emotion_adv,
        ).to(device=self.device)
        
        # Forward through T3
        out = self.t3.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )
        
        # Align logits (shift by 1)
        logits_for_text = out.text_logits[:, :-1, :].contiguous()
        logits_for_speech = out.speech_logits[:, :-1, :].contiguous()
        
        # Compute loss
        total_loss, loss_dict = self.loss_calculator(
            text_logits=logits_for_text,
            speech_logits=logits_for_speech,
            labels_text=labels_text,
            labels_speech=labels_speech,
        )
        
        # Store ALL loss components for detailed logging
        self.last_loss_dict = loss_dict
        self.last_loss_text = loss_dict.get("loss_text", torch.tensor(0.0))
        self.last_loss_speech = loss_dict.get("loss_speech", torch.tensor(0.0))
        self.last_z_loss = loss_dict.get("z_loss", torch.tensor(0.0))
        self.last_total_loss = loss_dict.get("total_loss", total_loss)
        
        return total_loss, out.speech_logits


class ResumeVerificationCallback(TrainerCallback):
    """
    Callback to verify and log checkpoint resume details.
    
    IMPORTANT: Replaces BestModelCallback. Use Trainer's built-in
    load_best_model_at_end=True instead for proper checkpoint management.
    
    This callback only logs resume information for debugging.
    """
    
    def __init__(self):
        self.has_logged_resume = False
    
    def on_train_begin(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl,
        **kwargs
    ):
        """Log training start or resume information."""
        if state.global_step > 0 and not self.has_logged_resume:
            # Resuming from checkpoint
            current_lr = self._get_current_lr(kwargs.get("optimizer"))
            
            logger.info("=" * 60)
            logger.info("🔄 RESUMING FROM CHECKPOINT")
            logger.info("=" * 60)
            logger.info(f"Step: {state.global_step}")
            logger.info(f"Epoch: {state.epoch:.2f}" if state.epoch is not None else "Epoch: N/A")
            logger.info(f"Learning Rate: {current_lr:.2e}" if current_lr else "Learning Rate: N/A")
            logger.info(f"Best Metric: {state.best_metric}" if state.best_metric is not None else "Best Metric: N/A")
            logger.info("=" * 60)
            
            self.has_logged_resume = True
            
            # Verify checkpoint integrity
            model = kwargs.get("model")
            if model is not None:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Model: {trainable_params:,} trainable / {total_params:,} total parameters")
        else:
            # Fresh start
            logger.info("✨ Starting fresh training")
            logger.info(f"Output directory: {args.output_dir}")
            logger.info(f"Logging steps: {args.logging_steps}")
            logger.info(f"Save steps: {args.save_steps}")
            logger.info(f"Eval steps: {args.eval_steps}" if args.eval_steps else "Eval: Disabled")
    
    def _get_current_lr(self, optimizer) -> Optional[float]:
        """Extract current learning rate from optimizer."""
        if optimizer is None:
            return None
        try:
            return optimizer.param_groups[0]["lr"]
        except:
            return None
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log checkpoint save events."""
        logger.info(f"💾 Checkpoint saved at step {state.global_step}")


class SafeCheckpointTrainer(Trainer):
    """
    Custom Trainer with:
    - Dynamic batching via LengthGroupedSampler
    - Safe checkpoint loading
    - Detailed loss logging for WandB
    """
    
    def __init__(
        self,
        *args,
        use_dynamic_batching: bool = True,
        bucket_size_multiplier: int = 100,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_dynamic_batching = use_dynamic_batching
        self.bucket_size_multiplier = bucket_size_multiplier
    
    def _get_train_sampler(self, train_dataset):
        """Override to use LengthGroupedSampler when enabled."""
        if not self.use_dynamic_batching:
            return super()._get_train_sampler(train_dataset)
        
        if train_dataset is None:
            return None
        
        logger.info("Using LengthGroupedSampler for dynamic batching")
        
        return LengthGroupedSampler(
            dataset=train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            seed=self.args.seed,
            bucket_size_multiplier=self.bucket_size_multiplier,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to track loss components for WandB."""
        # Move inputs to device
        inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Forward pass
        loss, outputs = model(**inputs)
        
        # Log ALL component losses to WandB
        if hasattr(model, "last_loss_dict") and model.last_loss_dict is not None:
            loss_dict = model.last_loss_dict
            
            if "loss_text" in loss_dict:
                self._log_custom_metric("train/loss_text", loss_dict["loss_text"])
            if "loss_speech" in loss_dict:
                self._log_custom_metric("train/loss_speech", loss_dict["loss_speech"])
            if "z_loss" in loss_dict:
                self._log_custom_metric("train/z_loss", loss_dict["z_loss"])
            if "total_loss" in loss_dict:
                self._log_custom_metric("train/total_loss", loss_dict["total_loss"])
            
            # Terminal logging every N steps
            if self.state.global_step % self.args.logging_steps == 0:
                loss_text_val = loss_dict.get("loss_text", 0.0)
                loss_speech_val = loss_dict.get("loss_speech", 0.0)
                z_loss_val = loss_dict.get("z_loss", 0.0)
                total_loss_val = loss_dict.get("total_loss", loss)
                
                if torch.is_tensor(loss_text_val):
                    loss_text_val = loss_text_val.item()
                if torch.is_tensor(loss_speech_val):
                    loss_speech_val = loss_speech_val.item()
                if torch.is_tensor(z_loss_val):
                    z_loss_val = z_loss_val.item()
                if torch.is_tensor(total_loss_val):
                    total_loss_val = total_loss_val.item()
                
                logger.info(
                    f"📊 Step {self.state.global_step} | "
                    f"Total: {total_loss_val:.4f} | "
                    f"Text: {loss_text_val:.4f} | "
                    f"Speech: {loss_speech_val:.4f} | "
                    f"ZLoss: {z_loss_val:.6f}"
                )
        
        return (loss, outputs) if return_outputs else loss
    
    def _log_custom_metric(self, name: str, value):
        """Helper to log custom metrics during training."""
        if self.state.global_step % self.args.logging_steps == 0:
            if hasattr(self, "log"):
                if torch.is_tensor(value):
                    value = value.item()
                self.log({name: value})
    
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Override to compute detailed eval metrics."""
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        # Add component losses to metrics
        model = self.model
        if hasattr(model, "last_loss_text"):
            loss_text = model.last_loss_text
            if torch.is_tensor(loss_text):
                loss_text = loss_text.item()
            output.metrics[f"{metric_key_prefix}_loss_text"] = loss_text
        
        if hasattr(model, "last_loss_speech"):
            loss_speech = model.last_loss_speech
            if torch.is_tensor(loss_speech):
                loss_speech = loss_speech.item()
            output.metrics[f"{metric_key_prefix}_loss_speech"] = loss_speech
        
        return output


def setup_wandb(project_name: str, run_name: Optional[str] = None, config: Optional[dict] = None):
    """
    Setup WandB for experiment tracking.
    
    Args:
        project_name: WandB project name
        run_name: Optional run name
        config: Optional config dict to log
    """
    try:
        import wandb
        
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            resume="allow",
        )
        logger.info(f"WandB initialized: project={project_name}, run={run_name}")
        return wandb
    except ImportError:
        logger.warning("WandB not installed. Install with: pip install wandb")
        return None
    except Exception as e:
        logger.warning(f"WandB init failed: {e}")
        return None
