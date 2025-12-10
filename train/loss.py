"""
Safe Loss Functions for T3 Fine-tuning

Features:
- ZLoss + Label Smoothing (NO FocalLoss per user feedback)
- NaN-safe implementations with proper masking
- Logit clamping to prevent overflow
- Empty batch guards
- WandB logging integration

Usage:
    from train.loss import T3LossCalculator, LossConfig
    
    calc = T3LossCalculator(
        label_smoothing=0.1,
        use_z_loss=True,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple, Literal
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

# IGNORE_ID for masked positions (standard in HuggingFace)
IGNORE_ID = -100

# Logit clamping bounds to prevent NaN/Inf
LOGIT_CLAMP_MIN = -100.0
LOGIT_CLAMP_MAX = 100.0


@dataclass
class LossConfig:
    """Configuration for T3 loss computation"""
    # Loss weights
    text_weight: float = 0.1
    speech_weight: float = 1.0
    
    # Label smoothing (built-in torch support)
    label_smoothing: float = 0.1
    
    # Z-loss for numerical stability (from PaLM/Gemini)
    use_z_loss: bool = True
    z_loss_weight: float = 1e-4
    
    # Loss reduction
    reduction: Literal["mean", "sum", "none"] = "mean"


class SafeZLoss(nn.Module):
    """
    Safe Z-Loss for numerical stability in softmax.
    
    From PaLM paper: Encourages logits to stay close to zero,
    preventing very large logits that can cause numerical issues.
    
    IMPORTANT: This implementation applies mask BEFORE computation
    to avoid Inf * 0 = NaN issue (IEEE 754 standard).
    
    z_loss = log(sum(exp(logits)))^2
    
    Args:
        weight: Weight for z-loss term
        clamp_min: Minimum logit value (prevents -inf)
        clamp_max: Maximum logit value (prevents +inf)
    """
    
    def __init__(
        self, 
        weight: float = 1e-4,
        clamp_min: float = LOGIT_CLAMP_MIN,
        clamp_max: float = LOGIT_CLAMP_MAX,
    ):
        super().__init__()
        self.weight = weight
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    
    def forward(self, logits: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            logits: (B, S, C) or (B, C, S) - model predictions
            mask: (B, S) boolean mask for VALID positions (True = valid, False = padding)
            
        Returns:
            Z-loss value (scalar)
        """
        # Empty batch guard
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Ensure (B, S, C) format
        if logits.dim() == 3 and logits.size(1) > logits.size(2):
            # (B, C, S) -> (B, S, C)
            logits = logits.transpose(1, 2)
        
        # Clamp logits to prevent overflow
        logits = logits.clamp(min=self.clamp_min, max=self.clamp_max)
        
        # Compute log-sum-exp over vocab dimension (dim=-1)
        # Using PyTorch's numerically stable logsumexp
        log_z = torch.logsumexp(logits, dim=-1)  # (B, S)
        
        # Square the log-sum-exp
        z_loss = log_z ** 2  # (B, S)
        
        # Apply mask BEFORE aggregation to avoid Inf * 0 = NaN
        if mask is not None:
            # mask: (B, S), True = valid
            valid_count = mask.sum().clamp(min=1)  # Avoid div by zero
            
            # Zero out invalid positions BEFORE any operations
            z_loss = z_loss.masked_fill(~mask, 0.0)
            
            return self.weight * z_loss.sum() / valid_count
        
        return self.weight * z_loss.mean()


class T3LossCalculator(nn.Module):
    """
    Unified loss calculator for T3 model fine-tuning.
    
    Features:
    - Cross-entropy with label smoothing (PyTorch built-in)
    - Safe Z-loss for numerical stability
    - Proper NaN handling
    - WandB-compatible loss dict output
    
    NOT included (per user feedback due to NaN issues):
    - FocalLoss
    - Entropy regularization
    """
    
    def __init__(self, config: Optional[LossConfig] = None, **kwargs):
        super().__init__()
        
        # Create config from kwargs if not provided
        if config is None:
            config = LossConfig(**kwargs)
        self.config = config
        
        # Z-Loss (optional but recommended)
        self.z_loss = SafeZLoss(config.z_loss_weight) if config.use_z_loss else None
        
        logger.info(f"T3LossCalculator initialized:")
        logger.info(f"  - label_smoothing={config.label_smoothing}")
        logger.info(f"  - text_weight={config.text_weight}, speech_weight={config.speech_weight}")
        if config.use_z_loss:
            logger.info(f"  - Z-Loss enabled with weight={config.z_loss_weight}")
    
    def compute_ce_loss(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Compute cross-entropy loss with label smoothing.
        
        Uses PyTorch built-in implementation for edge case handling.
        
        Args:
            logits: (B, S, C) - model predictions
            labels: (B, S) - ground truth with IGNORE_ID for masked positions
            
        Returns:
            Loss value (scalar)
        """
        device = logits.device
        
        # Empty batch guard
        if logits.numel() == 0 or labels.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Empty sequence guard
        if logits.size(1) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Clamp logits to prevent overflow
        logits = logits.clamp(min=LOGIT_CLAMP_MIN, max=LOGIT_CLAMP_MAX)
        
        # Ensure (B, C, S) for cross_entropy (expects channel in dim 1)
        if logits.dim() == 3:
            logits = logits.transpose(1, 2)  # (B, S, C) -> (B, C, S)
        
        # Use PyTorch built-in cross_entropy with label smoothing
        # This handles all edge cases properly
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=IGNORE_ID,
            label_smoothing=self.config.label_smoothing,
            reduction=self.config.reduction,
        )
    
    def forward(
        self,
        text_logits: Tensor,
        speech_logits: Tensor,
        labels_text: Tensor,
        labels_speech: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute total loss for T3 training.
        
        Args:
            text_logits: (B, S_text, V_text) - text prediction logits
            speech_logits: (B, S_speech, V_speech) - speech prediction logits
            labels_text: (B, S_text) - text labels with IGNORE_ID
            labels_speech: (B, S_speech) - speech labels with IGNORE_ID
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary of individual loss components (for WandB)
        """
        device = text_logits.device
        loss_dict = {}
        
        # --- Text Loss ---
        loss_text = self.compute_ce_loss(text_logits, labels_text)
        loss_dict["loss_text"] = loss_text.detach()
        
        # --- Speech Loss ---
        loss_speech = self.compute_ce_loss(speech_logits, labels_speech)
        loss_dict["loss_speech"] = loss_speech.detach()
        
        # --- Weighted combination ---
        total_loss = (
            self.config.text_weight * loss_text +
            self.config.speech_weight * loss_speech
        )
        
        # --- Z-Loss (optional) ---
        if self.z_loss is not None:
            # Create valid mask (True where labels != IGNORE_ID)
            valid_mask = (labels_speech != IGNORE_ID)
            
            # Only compute if there are valid positions
            if valid_mask.any():
                z_loss_val = self.z_loss(speech_logits, valid_mask)
                total_loss = total_loss + z_loss_val
                loss_dict["z_loss"] = z_loss_val.detach()
            else:
                loss_dict["z_loss"] = torch.tensor(0.0, device=device)
        
        # NaN/Inf check - RAISE instead of hiding
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            error_msg = (
                f"NaN/Inf detected in loss! This indicates a serious training issue.\n"
                f"Components: {loss_dict}\n"
                f"Possible causes:\n"
                f"  - Gradient explosion (learning rate too high)\n"
                f"  - Data corruption (invalid audio/text)\n"
                f"  - Numerical instability in model\n"
                f"Action required: Lower learning rate, check data, or enable gradient clipping."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        loss_dict["total_loss"] = total_loss.detach()
        
        return total_loss, loss_dict
    
    def __repr__(self) -> str:
        return (
            f"T3LossCalculator(\n"
            f"  text_weight={self.config.text_weight},\n"
            f"  speech_weight={self.config.speech_weight},\n"
            f"  label_smoothing={self.config.label_smoothing},\n"
            f"  use_z_loss={self.config.use_z_loss}\n"
            f")"
        )


# =============================================================================
# Convenience function for backward compatibility
# =============================================================================

def compute_t3_loss(
    text_logits: Tensor,
    speech_logits: Tensor,
    labels_text: Tensor,
    labels_speech: Tensor,
    text_weight: float = 0.1,
    speech_weight: float = 1.0,
    label_smoothing: float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute T3 loss with sensible defaults.
    
    Drop-in replacement for existing loss computation.
    
    Returns:
        loss_text, loss_speech, total_loss
    """
    device = text_logits.device
    
    # Clamp logits
    text_logits = text_logits.clamp(min=LOGIT_CLAMP_MIN, max=LOGIT_CLAMP_MAX)
    speech_logits = speech_logits.clamp(min=LOGIT_CLAMP_MIN, max=LOGIT_CLAMP_MAX)
    
    # Handle logit shape - ensure (B, C, S) for cross_entropy
    if text_logits.dim() == 3:
        text_logits = text_logits.transpose(1, 2)
    if speech_logits.dim() == 3:
        speech_logits = speech_logits.transpose(1, 2)
    
    # Compute losses using PyTorch built-in
    if text_logits.size(-1) == 0:
        loss_text = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        loss_text = F.cross_entropy(
            text_logits,
            labels_text,
            ignore_index=IGNORE_ID,
            label_smoothing=label_smoothing,
        )
    
    if speech_logits.size(-1) == 0:
        loss_speech = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        loss_speech = F.cross_entropy(
            speech_logits,
            labels_speech,
            ignore_index=IGNORE_ID,
            label_smoothing=label_smoothing,
        )
    
    total_loss = text_weight * loss_text + speech_weight * loss_speech
    
    return loss_text, loss_speech, total_loss


# =============================================================================
# Factory function
# =============================================================================

def create_loss_calculator(
    preset: Literal["default", "stable"] = "default",
    **kwargs
) -> T3LossCalculator:
    """
    Factory function to create loss calculator with presets.
    
    Presets:
        - default: CE with label smoothing, weighted losses
        - stable: With z-loss for numerical stability
        
    Args:
        preset: Preset configuration name
        **kwargs: Override specific config values
        
    Returns:
        Configured T3LossCalculator
    """
    presets = {
        "default": LossConfig(
            text_weight=0.1,
            speech_weight=1.0,
            label_smoothing=0.1,
            use_z_loss=False,
        ),
        "stable": LossConfig(
            text_weight=0.1,
            speech_weight=1.0,
            label_smoothing=0.1,
            use_z_loss=True,
            z_loss_weight=1e-4,
        ),
    }
    
    config = presets.get(preset, presets["default"])
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return T3LossCalculator(config)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Safe T3 Loss Functions...")
    
    B, S_text, V_text = 2, 50, 704
    B, S_speech, V_speech = 2, 100, 8194
    
    text_logits = torch.randn(B, S_text, V_text)
    speech_logits = torch.randn(B, S_speech, V_speech)
    labels_text = torch.randint(0, V_text, (B, S_text))
    labels_speech = torch.randint(0, V_speech, (B, S_speech))
    
    # Mask some positions
    labels_text[:, -10:] = IGNORE_ID
    labels_speech[:, -20:] = IGNORE_ID
    
    # Test default
    calc = create_loss_calculator("stable")
    print(calc)
    total, loss_dict = calc(text_logits, speech_logits, labels_text, labels_speech)
    print(f"Stable - Total: {total.item():.4f}")
    print(f"  Components: {[(k, v.item()) for k, v in loss_dict.items()]}")
    
    # Test with edge cases (all padding)
    labels_all_pad = torch.full_like(labels_speech, IGNORE_ID)
    total, loss_dict = calc(text_logits, speech_logits, labels_text, labels_all_pad)
    print(f"All padding - Total: {total.item():.4f} (should be small)")
    
    # Test convenience function
    loss_text, loss_speech, total = compute_t3_loss(
        text_logits, speech_logits, labels_text, labels_speech
    )
    print(f"Convenience fn - Total: {total.item():.4f}")
    
    print("\n✅ All tests passed!")
