import os
import torch
import logging
from pathlib import Path
from transformers import TrainerCallback, TrainerState, TrainerControl
import wandb
import numpy as np
import librosa
import soundfile as sf

from viterbox.tts import Viterbox
from .vietnamese_test_cases import TEST_CASES

logger = logging.getLogger(__name__)

class VietnameseEvalCallback(TrainerCallback):
    """
    Callback to generate audio for Vietnamese test cases during evaluation.
    """
    def __init__(
        self,
        viterbox_base: Viterbox,
        output_dir: str,
        upload_to_wandb: bool = True,
        max_samples_per_group: int = 2,  # Limit samples to save time
    ):
        self.viterbox_base = viterbox_base
        self.output_dir = Path(output_dir) / "eval_samples"
        self.upload_to_wandb = upload_to_wandb
        self.max_samples_per_group = max_samples_per_group
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Run generation on evaluation step"""
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer") # Not used, we have it in viterbox
        
        # Unwrap model if needed (e.g. DDP)
        if hasattr(model, "module"):
            model = model.module
            
        # Get the T3 model from usage wrapper
        if hasattr(model, "t3"):
            t3_model = model.t3
        else:
            logger.warning("⚠️ Could not find T3 model for evaluation generation")
            return

        # Create a temporary Viterbox instance for inference
        # We reuse S3Gen, VE, Tokenizer from the base, but use the current T3
        inference_model = Viterbox(
            t3=t3_model,
            s3gen=self.viterbox_base.s3gen,
            ve=self.viterbox_base.ve,
            tokenizer=self.viterbox_base.tokenizer,
            device=self.viterbox_base.device
        )
        
        # Reuse conditioning from base model if available for consistent voice
        if self.viterbox_base.conds is not None:
            inference_model.conds = self.viterbox_base.conds
        
        # Ensure models are in eval mode
        inference_model.t3.eval()
        inference_model.s3gen.eval()
        inference_model.ve.eval()
        
        step_dir = self.output_dir / f"step_{state.global_step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎤 Generating evaluation samples for step {state.global_step}...")
        
        try:
            # Generate samples for each group
            for group_name, prompts in TEST_CASES.items():
                group_clean = group_name.lower().replace(" ", "_").replace(":", "")
                group_dir = step_dir / group_clean
                group_dir.mkdir(exist_ok=True)
                
                # Take subset of prompts to save time
                selected_prompts = prompts[:self.max_samples_per_group]
                
                for i, text in enumerate(selected_prompts):
                    try:
                        # Generate
                        with torch.inference_mode():
                            # Use random voice if no prompt provided (or default one)
                            # Passing None to audio_prompt uses random/default from Viterbox
                            audio = inference_model.generate(
                                text=text,
                                language="vi",
                                audio_prompt=None, 
                                cfg_weight=0.7,  # Good default
                            )
                        
                        # Save audio
                        filename = f"{i}_{text[:30].replace(' ', '_')}.wav"
                        # Sanitize filename
                        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                        save_path = group_dir / filename
                        
                        inference_model.save_audio(audio, save_path)
                        
                        # Upload to WandB
                        if self.upload_to_wandb and wandb.run is not None:
                            wandb.log({
                                f"eval/{group_name}/{i}": wandb.Audio(
                                    str(save_path), 
                                    caption=text, 
                                    sample_rate=24000
                                )
                            }, commit=False)
                            
                    except Exception as e:
                        logger.error(f"Failed to generate '{text}': {e}")
            
            # Commit WandB logs
            if self.upload_to_wandb and wandb.run is not None:
                wandb.log({}, commit=True)
                
        except Exception as e:
            logger.error(f"Error during evaluation generation: {e}")
        finally:
            # Revert to training mode if needed (Trainer handles this mostly, but good practice)
            model.train()

