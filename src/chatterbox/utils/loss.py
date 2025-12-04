"""
Advanced Loss Functions for T3 Fine-tuning

This module provides SOTA loss functions compatible with Chatterbox T3 model training.
Prioritizes using built-in torch/transformers functions, custom implementations only when needed.

Usage:
    from chatterbox.utils.loss import T3LossCalculator
    
    loss_calc = T3LossCalculator(
        text_weight=0.1,
        speech_weight=1.0,
        label_smoothing=0.1,
        use_focal_loss=True,
    )
    
    total_loss, loss_dict = loss_calc(
        text_logits=...,
        speech_logits=...,
        labels_text=...,
        labels_speech=...,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# IGNORE_ID for masked positions (standard in HuggingFace)
IGNORE_ID = -100


@dataclass
class LossConfig:
    """Configuration for T3 loss computation"""
    # Loss weights
    text_weight: float = 0.1
    speech_weight: float = 1.0
    
    # Label smoothing (built-in torch support)
    label_smoothing: float = 0.1
    
    # Focal loss params (for class imbalance)
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    
    # Z-loss for numerical stability (from PaLM/Gemini)
    use_z_loss: bool = False
    z_loss_weight: float = 1e-4
    
    # Auxiliary losses
    use_entropy_regularization: bool = False
    entropy_weight: float = 0.01
    
    # Loss reduction
    reduction: Literal["mean", "sum", "none"] = "mean"


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in token classification.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Useful for Vietnamese TTS where certain tones/phonemes are rare.
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weight (optional)
        ignore_index: Index to ignore in loss computation
        label_smoothing: Label smoothing factor
        reduction: Loss reduction method
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        ignore_index: int = IGNORE_ID,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (B, C, S) or (B, S, C) - model predictions
            targets: (B, S) - ground truth labels
            
        Returns:
            Focal loss value
        """
        # Ensure logits are (B, C, S) format for cross_entropy
        if logits.dim() == 3 and logits.size(1) != logits.size(2):
            if logits.size(2) > logits.size(1):
                # (B, S, C) -> (B, C, S)
                logits = logits.transpose(1, 2)
        
        # Compute standard cross-entropy (per-element, no reduction)
        # Using built-in label_smoothing from PyTorch
        ce_loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )
        
        # Compute pt (probability of correct class)
        # Need to handle ignore_index properly
        valid_mask = (targets != self.ignore_index)
        
        # Get probabilities
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, S)
        
        # Gather correct class probabilities
        # targets: (B, S), need to handle ignore_index
        targets_for_gather = targets.clone()
        targets_for_gather[~valid_mask] = 0  # Replace ignore with valid index
        
        # (B, S)
        pt = torch.exp(-ce_loss)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == "mean":
            # Only average over valid (non-ignored) positions
            return focal_loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ZLoss(nn.Module):
    """
    Z-Loss for numerical stability in softmax.
    
    From PaLM paper: Encourages logits to stay close to zero,
    preventing very large logits that can cause numerical issues.
    
    z_loss = log(sum(exp(logits)))^2
    
    Args:
        weight: Weight for z-loss term
    """
    
    def __init__(self, weight: float = 1e-4):
        super().__init__()
        self.weight = weight
    
    def forward(self, logits: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            logits: (B, C, S) or (B, S, C)
            mask: (B, S) boolean mask for valid positions
            
        Returns:
            Z-loss value
        """
        # Compute log-sum-exp (numerically stable via logsumexp)
        if logits.dim() == 3:
            # Assume (B, C, S) or (B, S, C), we want logsumexp over vocab dim
            if logits.size(1) > logits.size(2):
                # (B, C, S) format
                log_z = torch.logsumexp(logits, dim=1)  # (B, S)
            else:
                # (B, S, C) format
                log_z = torch.logsumexp(logits, dim=2)  # (B, S)
        else:
            log_z = torch.logsumexp(logits, dim=-1)
        
        # Square the log-sum-exp
        z_loss = log_z ** 2
        
        # Apply mask if provided
        if mask is not None:
            z_loss = z_loss * mask.float()
            return self.weight * z_loss.sum() / mask.sum().clamp(min=1)
        
        return self.weight * z_loss.mean()


class EntropyRegularization(nn.Module):
    """
    Entropy regularization to encourage confident predictions.
    
    Can be used to:
    - Encourage low entropy (confident): positive weight
    - Encourage high entropy (diverse): negative weight
    
    Args:
        weight: Weight for entropy term (positive = penalize high entropy)
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, logits: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            logits: (B, S, C) or (B, C, S)
            mask: (B, S) boolean mask for valid positions
            
        Returns:
            Entropy regularization value
        """
        # Get probabilities
        if logits.dim() == 3 and logits.size(1) > logits.size(2):
            # (B, C, S) -> (B, S, C)
            logits = logits.transpose(1, 2)
        
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Entropy: -sum(p * log(p))
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, S)
        
        # Apply mask
        if mask is not None:
            entropy = entropy * mask.float()
            return self.weight * entropy.sum() / mask.sum().clamp(min=1)
        
        return self.weight * entropy.mean()


class T3LossCalculator(nn.Module):
    """
    Unified loss calculator for T3 model fine-tuning.
    
    Combines:
    - Cross-entropy with label smoothing (PyTorch built-in)
    - Optional focal loss for class imbalance
    - Optional z-loss for numerical stability
    - Optional entropy regularization
    - Configurable text/speech loss weighting
    
    Compatible with existing T3 training pipeline.
    """
    
    def __init__(self, config: Optional[LossConfig] = None, **kwargs):
        super().__init__()
        
        # Create config from kwargs if not provided
        if config is None:
            config = LossConfig(**kwargs)
        self.config = config
        
        # Initialize loss functions
        if config.use_focal_loss:
            self.text_loss_fn = FocalLoss(
                gamma=config.focal_gamma,
                alpha=config.focal_alpha,
                label_smoothing=config.label_smoothing,
                reduction=config.reduction,
            )
            self.speech_loss_fn = FocalLoss(
                gamma=config.focal_gamma,
                alpha=config.focal_alpha,
                label_smoothing=config.label_smoothing,
                reduction=config.reduction,
            )
            logger.info(f"Using FocalLoss with gamma={config.focal_gamma}")
        else:
            # Use standard cross-entropy with label smoothing (PyTorch built-in)
            self.text_loss_fn = None
            self.speech_loss_fn = None
            logger.info(f"Using CrossEntropyLoss with label_smoothing={config.label_smoothing}")
        
        # Auxiliary losses
        self.z_loss = ZLoss(config.z_loss_weight) if config.use_z_loss else None
        self.entropy_reg = EntropyRegularization(config.entropy_weight) if config.use_entropy_regularization else None
        
        if config.use_z_loss:
            logger.info(f"Using Z-Loss with weight={config.z_loss_weight}")
        if config.use_entropy_regularization:
            logger.info(f"Using Entropy Regularization with weight={config.entropy_weight}")
    
    def compute_ce_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        use_focal: bool = False,
        loss_fn: Optional[nn.Module] = None,
    ) -> Tensor:
        """
        Compute cross-entropy loss with optional enhancements.
        
        Args:
            logits: (B, S, C) or (B, C, S)
            labels: (B, S) with IGNORE_ID for masked positions
            use_focal: Whether to use focal loss
            loss_fn: Optional custom loss function
            
        Returns:
            Loss value
        """
        device = logits.device
        
        # Handle empty sequences
        if logits.size(1) == 0 or logits.size(2) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Ensure logits are (B, C, S) for cross_entropy
        if logits.dim() == 3:
            # Check if we need to transpose
            # Typically: (B, S, C) where S < C for speech/text tokens
            if logits.size(1) < logits.size(2):
                # Already (B, S, C) or small S -> transpose to (B, C, S)
                logits = logits.transpose(1, 2)
        
        if use_focal and loss_fn is not None:
            return loss_fn(logits, labels)
        else:
            # Use built-in PyTorch cross_entropy with label smoothing
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
        return_components: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute total loss for T3 training.
        
        Args:
            text_logits: (B, S_text, V_text) - text prediction logits
            speech_logits: (B, S_speech, V_speech) - speech prediction logits
            labels_text: (B, S_text) - text labels with IGNORE_ID
            labels_speech: (B, S_speech) - speech labels with IGNORE_ID
            return_components: Whether to return individual loss components
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary of individual loss components
        """
        device = text_logits.device
        loss_dict = {}
        
        # --- Text Loss ---
        loss_text = self.compute_ce_loss(
            text_logits,
            labels_text,
            use_focal=self.config.use_focal_loss,
            loss_fn=self.text_loss_fn,
        )
        loss_dict["loss_text"] = loss_text
        
        # --- Speech Loss ---
        loss_speech = self.compute_ce_loss(
            speech_logits,
            labels_speech,
            use_focal=self.config.use_focal_loss,
            loss_fn=self.speech_loss_fn,
        )
        loss_dict["loss_speech"] = loss_speech
        
        # --- Weighted combination ---
        total_loss = (
            self.config.text_weight * loss_text +
            self.config.speech_weight * loss_speech
        )
        
        # --- Auxiliary losses ---
        if self.z_loss is not None:
            # Compute z-loss on speech logits (more important for TTS)
            valid_mask = (labels_speech != IGNORE_ID)
            z_loss_val = self.z_loss(speech_logits, valid_mask)
            total_loss = total_loss + z_loss_val
            loss_dict["z_loss"] = z_loss_val
        
        if self.entropy_reg is not None:
            valid_mask = (labels_speech != IGNORE_ID)
            entropy_val = self.entropy_reg(speech_logits, valid_mask)
            total_loss = total_loss + entropy_val
            loss_dict["entropy_reg"] = entropy_val
        
        loss_dict["total_loss"] = total_loss
        
        return total_loss, loss_dict
    
    def __repr__(self) -> str:
        return (
            f"T3LossCalculator(\n"
            f"  text_weight={self.config.text_weight},\n"
            f"  speech_weight={self.config.speech_weight},\n"
            f"  label_smoothing={self.config.label_smoothing},\n"
            f"  use_focal_loss={self.config.use_focal_loss},\n"
            f"  use_z_loss={self.config.use_z_loss},\n"
            f"  use_entropy_regularization={self.config.use_entropy_regularization}\n"
            f")"
        )


# =============================================================================
# Convenience functions for backward compatibility
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
    
    Drop-in replacement for existing loss computation in t3.py
    
    Args:
        text_logits: (B, S_text, V_text) or (B, V_text, S_text)
        speech_logits: (B, S_speech, V_speech) or (B, V_speech, S_speech)
        labels_text: (B, S_text)
        labels_speech: (B, S_speech)
        text_weight: Weight for text loss
        speech_weight: Weight for speech loss
        label_smoothing: Label smoothing factor
        
    Returns:
        loss_text, loss_speech, total_loss
    """
    device = text_logits.device
    
    # Handle logit shape - ensure (B, C, S) for cross_entropy
    if text_logits.dim() == 3 and text_logits.size(1) < text_logits.size(2):
        text_logits = text_logits.transpose(1, 2)
    if speech_logits.dim() == 3 and speech_logits.size(1) < speech_logits.size(2):
        speech_logits = speech_logits.transpose(1, 2)
    
    # Compute losses using PyTorch built-in
    if text_logits.size(1) == 0:
        loss_text = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        loss_text = F.cross_entropy(
            text_logits,
            labels_text,
            ignore_index=IGNORE_ID,
            label_smoothing=label_smoothing,
        )
    
    if speech_logits.size(1) == 0:
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
# Factory function for creating loss calculators
# =============================================================================

def create_loss_calculator(
    preset: Literal["default", "focal", "stable", "aggressive"] = "default",
    **kwargs
) -> T3LossCalculator:
    """
    Factory function to create loss calculator with presets.
    
    Presets:
        - default: Standard CE with label smoothing, weighted losses
        - focal: Focal loss for handling rare tokens (good for Vietnamese tones)
        - stable: With z-loss for numerical stability (long training)
        - aggressive: All bells and whistles
        
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
            use_focal_loss=False,
            use_z_loss=False,
        ),
        "focal": LossConfig(
            text_weight=0.1,
            speech_weight=1.0,
            label_smoothing=0.05,
            use_focal_loss=True,
            focal_gamma=2.0,
            use_z_loss=False,
        ),
        "stable": LossConfig(
            text_weight=0.1,
            speech_weight=1.0,
            label_smoothing=0.1,
            use_focal_loss=False,
            use_z_loss=True,
            z_loss_weight=1e-4,
        ),
        "aggressive": LossConfig(
            text_weight=0.1,
            speech_weight=1.0,
            label_smoothing=0.1,
            use_focal_loss=True,
            focal_gamma=2.0,
            use_z_loss=True,
            z_loss_weight=1e-4,
            use_entropy_regularization=True,
            entropy_weight=0.01,
        ),
    }
    
    config = presets.get(preset, presets["default"])
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return T3LossCalculator(config)


# =============================================================================
# Test / Validation
# =============================================================================

if __name__ == "__main__":
    # Quick test
    print("Testing T3 Loss Functions...")
    
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
    calc = create_loss_calculator("default")
    print(calc)
    total, loss_dict = calc(text_logits, speech_logits, labels_text, labels_speech)
    print(f"Default - Total: {total.item():.4f}, Components: {[(k, v.item()) for k, v in loss_dict.items()]}")
    
    # Test focal
    calc = create_loss_calculator("focal")
    total, loss_dict = calc(text_logits, speech_logits, labels_text, labels_speech)
    print(f"Focal - Total: {total.item():.4f}")
    
    # Test stable
    calc = create_loss_calculator("stable")
    total, loss_dict = calc(text_logits, speech_logits, labels_text, labels_speech)
    print(f"Stable - Total: {total.item():.4f}, Z-Loss: {loss_dict.get('z_loss', 0)}")
    
    # Test convenience function
    loss_text, loss_speech, total = compute_t3_loss(
        text_logits, speech_logits, labels_text, labels_speech
    )
    print(f"Convenience fn - Text: {loss_text.item():.4f}, Speech: {loss_speech.item():.4f}, Total: {total.item():.4f}")
    
    print("\n✅ All tests passed!")
