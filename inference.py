"""
Viterbox - Command Line Inference
"""
import argparse
from pathlib import Path
from viterbox import Viterbox


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
        group_clean = group_name.lower().replace(" ", "_").replace(":", "")
        group_dir = out_path / group_clean
        group_dir.mkdir(exist_ok=True)
        
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
                safe_text = "".join(c for c in text[:30] if c.isalnum() or c in "._- ")
                safe_text = safe_text.replace(" ", "_")
                filename = f"{i}_{safe_text}.wav"
                
                tts.save_audio(audio, group_dir / filename)
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ All tests completed! Results saved to: {output_dir}")


def load_model_hybrid(device: str, model_path: str = None) -> Viterbox:
    """
    Load model with hybrid strategy:
    1. Load base components (S3Gen, VE, Tokenizer) from HuggingFace
    2. If model_path provided:
       - Load T3 weights from checkpoint
       - Apply LoRA adapters if present
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
        
        # Case 1: LoRA Adapter
        if (ckpt_path / "adapter_config.json").exists():
            print("  ✨ Detected LoRA adapter")
            try:
                from peft import PeftModel, PeftConfig
                
                # Load configuration
                config = PeftConfig.from_pretrained(str(ckpt_path))
                
                # We need to wrap the internal transformer (t3.tfmr)
                # Note: viterbox.tts.py loads T3, which has .tfmr (LlamaModel)
                
                # Apply LoRA
                print("  Applying LoRA adapters to T3 backbone...")
                tts.t3.tfmr = PeftModel.from_pretrained(tts.t3.tfmr, str(ckpt_path))
                tts.t3.to(device)
                
                print("  ✅ LoRA adapters merged successfully")
            except ImportError:
                print("  ⚠️ PEFT library not found. Cannot load LoRA adapter.")
            except Exception as e:
                print(f"  ❌ Failed to load LoRA: {e}")
                
        # Case 2: Full Finetune (safetensors)
        # Check for model.safetensors or t3_model.safetensors or pytorch_model.bin
        elif (ckpt_path / "model.safetensors").exists() or \
             (ckpt_path / "t3_model.safetensors").exists() or \
             (ckpt_path / "pytorch_model.bin").exists():
             
            print("  📦 Detected full fine-tuned weights")
            
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
                
                if len(unexpected) > 0:
                    print(f"  ⚠️ Unexpected keys: {unexpected[:5]}...")
                
                print(f"  ✅ Loaded weights from {w_path.name}")
                
            except Exception as e:
                print(f"  ❌ Failed to load weights: {e}")
        
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
