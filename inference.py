"""
Viterbox - Command Line Inference
"""
import argparse
from pathlib import Path
from viterbox import Viterbox


def _sanitize_path(name: str) -> str:
    """
    Sanitize a string to be safe for use as directory/file name.
    Removes or replaces characters that are problematic for file systems.
    """
    import re
    # Replace / and \ with underscore
    name = name.replace("/", "_").replace("\\", "_")
    # Replace : with empty (for "Group 1:" style names)
    name = name.replace(":", "")
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove any other non-alphanumeric characters except _ and -
    name = re.sub(r'[^\w\-]', '', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    # Lowercase
    name = name.lower()
    return name


def run_test_cases(tts: Viterbox, output_dir: str, ref_audio: str = None):
    """Run all test cases from vietnamese_test_cases.py"""
    try:
        from train.vietnamese_test_cases import TEST_CASES
    except ImportError:
        # If running from root without package install
        import sys
        sys.path.append(str(Path.cwd()))
        from train.vietnamese_test_cases import TEST_CASES

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Running {len(TEST_CASES)} test groups...")
    
    for group_name, prompts in TEST_CASES.items():
        # Properly sanitize group name to avoid path issues
        group_clean = _sanitize_path(group_name)
        group_dir = out_path / group_clean
        group_dir.mkdir(parents=True, exist_ok=True)  # parents=True for nested paths
        
        print(f"\n📂 Group: {group_name}")
        
        for i, text in enumerate(prompts):
            print(f"  generating: {text[:40]}...")
            try:
                audio = tts.generate(
                    text=text,
                    language="vi",
                    audio_prompt=ref_audio,
                    cfg_weight=0.7,
                    temperature=0.1,
                )
                
                # Sanitize filename
                safe_text = _sanitize_path(text[:30])
                filename = f"{i}_{safe_text}.wav"
                
                tts.save_audio(audio, group_dir / filename)
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ All tests completed! Results saved to: {output_dir}")


def _is_lora_checkpoint(state_dict: dict) -> bool:
    """
    Check if state_dict contains LoRA weights.
    LoRA keys have patterns like: 'lora_A.default.weight', 'lora_B.default.weight', 'base_layer.weight'
    """
    for key in state_dict.keys():
        if 'lora_A' in key or 'lora_B' in key or 'base_layer' in key:
            return True
    return False


def _merge_lora_weights(state_dict: dict, lora_alpha: int = 16, lora_r: int = 8) -> dict:
    """
    Merge LoRA weights (base_layer + lora_A @ lora_B * scaling) into standard weights.
    
    LoRA formula: W = W_base + (lora_B @ lora_A) * (alpha / r)
    
    Args:
        state_dict: State dict with LoRA keys
        lora_alpha: LoRA alpha (scaling numerator), default 16
        lora_r: LoRA rank, default 8
    
    Returns:
        Merged state dict with standard weight names
    """
    import re
    
    merged = {}
    lora_groups = {}  # Group by base key: {base_key: {base, lora_A, lora_B}}
    
    scaling = lora_alpha / lora_r
    
    for key, value in state_dict.items():
        # Identify LoRA-related keys
        # Pattern: some.module.base_layer.weight, some.module.lora_A.default.weight
        
        if '.base_layer.' in key:
            # Extract base key: some.module.base_layer.weight -> some.module
            base_key = key.replace('.base_layer.weight', '').replace('.base_layer.bias', '')
            suffix = 'weight' if 'weight' in key else 'bias'
            
            if base_key not in lora_groups:
                lora_groups[base_key] = {}
            lora_groups[base_key][f'base_{suffix}'] = value
            
        elif '.lora_A.' in key:
            # some.module.lora_A.default.weight -> some.module
            base_key = re.sub(r'\.lora_A\.[^.]+\.weight$', '', key)
            if base_key not in lora_groups:
                lora_groups[base_key] = {}
            lora_groups[base_key]['lora_A'] = value
            
        elif '.lora_B.' in key:
            # some.module.lora_B.default.weight -> some.module
            base_key = re.sub(r'\.lora_B\.[^.]+\.weight$', '', key)
            if base_key not in lora_groups:
                lora_groups[base_key] = {}
            lora_groups[base_key]['lora_B'] = value
            
        else:
            # Non-LoRA key, keep as-is
            merged[key] = value
    
    # Now merge each LoRA group
    for base_key, components in lora_groups.items():
        if 'base_weight' in components:
            base_weight = components['base_weight']
            
            # Check if we have LoRA components
            if 'lora_A' in components and 'lora_B' in components:
                lora_A = components['lora_A']  # Shape: (r, in_features)
                lora_B = components['lora_B']  # Shape: (out_features, r)
                
                # Merge: W = W_base + (B @ A) * scaling
                delta = (lora_B @ lora_A) * scaling
                merged_weight = base_weight + delta
                
                merged[f'{base_key}.weight'] = merged_weight
            else:
                # No LoRA, just use base
                merged[f'{base_key}.weight'] = base_weight
        
        if 'base_bias' in components:
            merged[f'{base_key}.bias'] = components['base_bias']
    
    return merged


def load_model_hybrid(device: str, model_path: str = None) -> Viterbox:
    """
    Load model with hybrid strategy:
    1. Load base components (S3Gen, VE, Tokenizer) from HuggingFace
    2. If model_path provided:
       - Load T3 weights from checkpoint
       - Apply LoRA adapters if present (auto-detect and merge)
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors
    import torch
    
    print("📥 Loading base components from HuggingFace Hub...")
    base_ckpt_dir = Path(
        snapshot_download(
            repo_id="dolly-vn/viterbox",
            repo_type="model",
            revision="main",
            allow_patterns=[
                "ve.pt",
                "s3gen.pt",
                "tokenizer_vi_expanded.json",
                "conds.pt",
                "t3_ml24ls_v2.safetensors", # Fallback base T3
            ],
        )
    )
    
    # Load base model first
    tts = Viterbox.from_local(base_ckpt_dir, device=device)
    
    if model_path:
        ckpt_path = Path(model_path)
        print(f"🔄 Overriding with local checkpoint: {ckpt_path}")
        
        # Case 1: LoRA Adapter (PRIORITY - check first!)
        if (ckpt_path / "adapter_config.json").exists():
            print("  ✨ Detected LoRA adapter (adapter_config.json)")
            try:
                from peft import PeftModel, PeftConfig
                
                # Load configuration
                config = PeftConfig.from_pretrained(str(ckpt_path))
                
                # Apply LoRA to the transformer backbone
                print("  Applying LoRA adapters to T3 backbone...")
                tts.t3.tfmr = PeftModel.from_pretrained(tts.t3.tfmr, str(ckpt_path))
                tts.t3.to(device)
                
                print("  ✅ LoRA adapters loaded successfully")
                
                # CRITICAL FIX: Early return to prevent loading model.safetensors
                # This is the key fix - without this, code falls through to Case 2
                # and loads base weights instead of LoRA weights!
                return tts
                
            except ImportError:
                print("  ⚠️ PEFT library not found. Cannot load LoRA adapter.")
                print("  Attempting to fall back to full weights...")
            except Exception as e:
                print(f"  ❌ Failed to load LoRA adapter: {e}")
                print("  Attempting to fall back to full weights...")
                import traceback
                traceback.print_exc()
                
        # Case 2: Full Finetune or Merged LoRA (only if LoRA loading failed or not present)
        elif (ckpt_path / "model.safetensors").exists() or \
             (ckpt_path / "t3_model.safetensors").exists() or \
             (ckpt_path / "pytorch_model.bin").exists():
             
            # Find the weight file
            if (ckpt_path / "model.safetensors").exists():
                w_path = ckpt_path / "model.safetensors"
            elif (ckpt_path / "t3_model.safetensors").exists():
                w_path = ckpt_path / "t3_model.safetensors"
            else:
                w_path = ckpt_path / "pytorch_model.bin"
                
            try:
                if str(w_path).endswith(".safetensors"):
                    state_dict = load_safetensors(w_path)
                else:
                    state_dict = torch.load(w_path, map_location="cpu", weights_only=True)
                
                # Check if this is a LoRA checkpoint (has base_layer/lora_A/lora_B)
                if _is_lora_checkpoint(state_dict):
                    print("  ✨ Detected LoRA weights in checkpoint (auto-merging)")
                    
                    # Try to read lora config from adapter_config.json or use defaults
                    lora_alpha, lora_r = 16, 8
                    try:
                        import json
                        adapter_cfg_path = ckpt_path / "adapter_config.json"
                        if adapter_cfg_path.exists():
                            with open(adapter_cfg_path) as f:
                                cfg = json.load(f)
                                lora_alpha = cfg.get("lora_alpha", 16)
                                lora_r = cfg.get("r", 8)
                    except:
                        pass
                    
                    print(f"  Using LoRA config: alpha={lora_alpha}, r={lora_r}")
                    state_dict = _merge_lora_weights(state_dict, lora_alpha=lora_alpha, lora_r=lora_r)
                    print(f"  ✅ Merged LoRA weights successfully")
                else:
                    print("  📦 Detected full fine-tuned weights")
                
                # Handle prefixes from Trainer (e.g. "t3.tfmr..." -> "tfmr...")
                # The Viterbox T3 model expects keys relative to T3 class
                
                # Helper to clean keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Remove "module." if from DDP
                    if k.startswith("module."):
                        k = k[7:]
                    
                    # Remove "t3." prefix if saved from Viterbox wrapper
                    if k.startswith("t3."):
                        k = k[3:]
                    
                    # Handle PEFT/LoRA wrappers (base_model.model)
                    # Mapping: tfmr.base_model.model.X -> tfmr.X
                    if "base_model.model." in k:
                        k = k.replace("base_model.model.", "")
                    
                    new_state_dict[k] = v
                
                # Load into T3
                missing, unexpected = tts.t3.load_state_dict(new_state_dict, strict=False)
                
                # Validate loading
                if len(missing) > 0 and len(missing) < 10:
                    print(f"  ⚠️ Missing keys: {missing}")
                elif len(missing) >= 10:
                    print(f"  ⚠️ Missing {len(missing)} keys (showing first 5): {missing[:5]}...")
                
                if len(unexpected) > 0:
                    print(f"  ⚠️ Unexpected keys ({len(unexpected)} total): {unexpected[:5]}...")
                
                # Move model to device and set to eval
                tts.t3.to(device).eval()
                
                # Verify model is on correct device
                print(f"  📍 Model device: {tts.t3.device}")
                print(f"  📍 Text embeddings shape: {tts.t3.text_emb.weight.shape}")
                print(f"  📍 Speech embeddings shape: {tts.t3.speech_emb.weight.shape}")
                
                # Test forward pass with dummy input
                try:
                    with torch.no_grad():
                        dummy_text = torch.randint(0, 100, (1, 10), device=device)
                        dummy_speech = torch.randint(0, 100, (1, 10), device=device)
                        
                        # Try embedding lookup
                        test_emb = tts.t3.text_emb(dummy_text)
                        print(f"  ✅ Text embedding test passed, shape: {test_emb.shape}")
                        
                except Exception as e:
                    print(f"  ❌ Model validation failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"  ✅ Loaded weights from {w_path.name}")
                
            except Exception as e:
                import traceback
                print(f"  ❌ Failed to load weights: {e}")
                traceback.print_exc()
        
    return tts


def main():
    parser = argparse.ArgumentParser(description="Viterbox Text-to-Speech")
    parser.add_argument("--model_path", "-m", type=str, default=None, help="Path to local checkpoint (LoRA or Full)")
    parser.add_argument("--text", "-t", type=str, help="Text to synthesize (ignored if --test-cases is used)")
    parser.add_argument("--test-cases", action="store_true", help="Run full Vietnamese test suite instead of single text")
    parser.add_argument("--lang", "-l", type=str, default="vi", help="Language (vi/en)")
    parser.add_argument("--ref", "-r", type=str, default=None, help="Reference audio for voice cloning (wav file)")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output file/dir path")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device (cuda/cpu/mps)")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Expression intensity (0.0-2.0)")
    parser.add_argument("--cfg-weight", type=float, default=0.7, help="CFG weight (0.0-1.0), higher is more stable")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (0.1-1.0), lower is more stable")
    parser.add_argument("--sentence-pause", type=float, default=0.5, help="Pause between sentences in seconds (default 0.5)")
    
    args = parser.parse_args()
    
    if not args.text and not args.test_cases:
        parser.error("Either --text or --test-cases must be provided")
    
    print(f"Loading model on {args.device}...")
    try:
        tts = load_model_hybrid(args.device, args.model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check reference audio
    if args.ref:
        ref_path = Path(args.ref)
        if not ref_path.exists():
            print(f"❌ Reference audio not found: {args.ref}")
            return
            
    if args.test_cases:
        output_dir = args.output if args.output != "output.wav" else "test_results"
        run_test_cases(tts, output_dir, args.ref)
        return
    
    print(f"Generating: '{args.text}'")
    
    try:
        audio = tts.generate(
            text=args.text,
            language=args.lang,
            audio_prompt=args.ref,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            sentence_pause_ms=int(args.sentence_pause * 1000),
        )
        
        tts.save_audio(audio, args.output)
        print(f"✅ Saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    main()
