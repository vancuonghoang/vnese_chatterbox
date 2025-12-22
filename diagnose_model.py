"""
Diagnostic script to check model compatibility issues
"""
import torch
from pathlib import Path
import sys

def check_model_compatibility(model_path: str = None):
    """Check if LoRA model is compatible with base model"""
    print("="*60)
    print("🔍 MODEL COMPATIBILITY DIAGNOSTIC")
    print("="*60)
    
    # Load base model
    print("\n1️⃣ Loading base model...")
    try:
        from viterbox import Viterbox
        tts = Viterbox.from_pretrained("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   ✅ Base model loaded")
    except Exception as e:
        print(f"   ❌ Failed to load base model: {e}")
        return
    
    # Check base model specs
    print("\n2️⃣ Base model specifications:")
    print(f"   Tokenizer vocab size: {tts.tokenizer.text_dict_size}")
    print(f"   T3 embedding size: {tts.t3.tfmr.embed_tokens.num_embeddings}")
    print(f"   T3 model type: {type(tts.t3.tfmr).__name__}")
    print(f"   S3Gen model type: {type(tts.s3gen).__name__}")
    
    # Check if vocab sizes match
    if tts.tokenizer.text_dict_size != tts.t3.tfmr.embed_tokens.num_embeddings:
        print(f"   ⚠️  WARNING: Tokenizer vocab ({tts.tokenizer.text_dict_size}) != "
              f"T3 embedding ({tts.t3.tfmr.embed_tokens.num_embeddings})")
    else:
        print(f"   ✅ Vocab sizes match")
    
    # Test base model generation
    print("\n3️⃣ Testing base model generation...")
    try:
        # Simple Vietnamese text
        test_audio = tts.generate(
            text="Xin chào",
            language="vi",
            cfg_weight=0.7,
            temperature=0.1,
        )
        if test_audio is not None and test_audio.numel() > 0:
            print(f"   ✅ Base model works! Generated {test_audio.numel()} samples")
        else:
            print(f"   ❌ Base model returned empty audio")
    except Exception as e:
        print(f"   ❌ Base model generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # If model_path provided, check LoRA compatibility
    if model_path:
        print(f"\n4️⃣ Checking LoRA checkpoint: {model_path}")
        ckpt_path = Path(model_path)
        
        # Check if LoRA
        if (ckpt_path / "adapter_config.json").exists():
            print("   📦 Detected: LoRA adapter")
            
            # Check config
            import json
            with open(ckpt_path / "adapter_config.json") as f:
                config = json.load(f)
            print(f"   LoRA rank: {config.get('r', 'N/A')}")
            print(f"   LoRA alpha: {config.get('lora_alpha', 'N/A')}")
            print(f"   Target modules: {config.get('target_modules', 'N/A')}")
            print(f"   Base model: {config.get('base_model_name_or_path', 'N/A')}")
            
        elif (ckpt_path / "model.safetensors").exists():
            print("   📦 Detected: Full fine-tuned model")
            
            # Check vocab size in checkpoint
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path / "model.safetensors")
            
            # Look for embedding weights
            embedding_keys = [k for k in state_dict.keys() if 'embed_tokens' in k or 'embeddings' in k]
            if embedding_keys:
                emb_key = embedding_keys[0]
                emb_size = state_dict[emb_key].shape[0]
                print(f"   Checkpoint embedding size: {emb_size}")
                
                if emb_size != tts.tokenizer.text_dict_size:
                    print(f"   ⚠️  MISMATCH: Checkpoint ({emb_size}) vs Tokenizer ({tts.tokenizer.text_dict_size})")
                    print(f"   💡 Solution: Load matching tokenizer from checkpoint")
                else:
                    print(f"   ✅ Embedding size matches tokenizer")
        
        # Check if tokenizer exists in checkpoint
        tokenizer_files = list(ckpt_path.glob("tokenizer*.json"))
        if tokenizer_files:
            print(f"   ✅ Found tokenizer: {[f.name for f in tokenizer_files]}")
        else:
            print(f"   ⚠️  No tokenizer found in checkpoint")
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    if not model_path:
        print("ℹ️  Run with --model_path to check LoRA compatibility")
    
    print("\n💡 Common issues:")
    print("   1. Vocab mismatch → Load tokenizer from checkpoint")
    print("   2. LoRA rank mismatch → Retrain with matching rank")
    print("   3. Base model version → Use same version for training/inference")
    print("   4. S3Gen weights missing → Check if s3gen.pt exists")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default=None)
    args = parser.parse_args()
    
    check_model_compatibility(args.model_path)
