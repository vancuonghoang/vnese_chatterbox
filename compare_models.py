#!/usr/bin/env python3
"""
Compare Base Model vs LoRA-Trained Model using Vietnamese Test Cases
Optimized for MacBook M1 (uses MPS device)

Usage:
    python compare_models.py \
        --lora_checkpoint ./checkpoints/checkpoint-1000 \
        --output_dir ./comparison \
        --ref_audio ./wavs/sample.wav
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from viterbox import Viterbox
from train.vietnamese_test_cases import TEST_CASES


def generate_for_test_cases(model, model_name: str, output_dir: Path, ref_audio=None):
    """Generate audio for all test cases"""
    
    for group_name, prompts in TEST_CASES.items():
        group_clean = group_name.lower().replace(" ", "_").replace(":", "")
        group_dir = output_dir / model_name / group_clean
        group_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n   📂 {group_name} ({len(prompts)} tests)")
        
        for i, text in enumerate(prompts):
            try:
                # Generate audio
                audio = model.generate(
                    text=text,
                    language="vi",
                    audio_prompt=ref_audio,
                    cfg_weight=0.7,
                    temperature=0.1
                )
                
                # Sanitize filename
                safe_text = "".join(c for c in text[:30] if c.isalnum() or c in "._- ")
                safe_text = safe_text.replace(" ", "_")
                filename = f"{i}_{safe_text}.wav"
                
                # Save
                save_path = group_dir / filename
                model.save_audio(audio, save_path)
                
                print(f"      ✓ {i}: {text[:40]}...")
                
            except Exception as e:
                print(f"      ✗ {i}: Failed - {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare base vs LoRA-trained model with test cases")
    parser.add_argument("--lora_checkpoint", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", default="./comparison", help="Output directory")
    parser.add_argument("--ref_audio", default=None, help="Reference audio for voice cloning")
    parser.add_argument("--skip_base", action="store_true", help="Skip base model (only test LoRA)")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect device (M1 Mac support)
    if torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🎮 Using CUDA GPU")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON: Base vs LoRA on Vietnamese Test Suite")
    print(f"{'='*70}")
    print(f"\nTotal test groups: {len(TEST_CASES)}")
    print(f"Total test cases: {sum(len(v) for v in TEST_CASES.values())}")
    
    # ========== BASE MODEL ==========
    if not args.skip_base:
        print(f"\n{'='*70}")
        print("1️⃣  BASE MODEL (Pretrained)")
        print(f"{'='*70}")
        
        try:
            base_model = Viterbox.from_pretrained(device)
            print("✅ Base model loaded\n")
            
            generate_for_test_cases(base_model, "base_model", out_dir, args.ref_audio)
            
        except Exception as e:
            print(f"❌ Base model failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # ========== LORA MODEL ==========
    print(f"\n{'='*70}")
    print("2️⃣  LoRA MODEL (Fine-tuned)")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.lora_checkpoint}\n")
    
    try:
        from inference import load_model_hybrid
        
        lora_model = load_model_hybrid(device, args.lora_checkpoint)
        print("✅ LoRA model loaded\n")
        
        generate_for_test_cases(lora_model, "lora_model", out_dir, args.ref_audio)
        
    except Exception as e:
        print(f"❌ LoRA model failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== SUMMARY ==========
    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Output: {out_dir.absolute()}")
    print(f"\n📊 Structure:")
    print(f"   comparison/")
    if not args.skip_base:
        print(f"   ├── base_model/")
        print(f"   │   ├── group_1_tone_matrix/")
        print(f"   │   ├── group_2_vowel_distinction/")
        print(f"   │   └── ...")
    print(f"   └── lora_model/")
    print(f"       ├── group_1_tone_matrix/")
    print(f"       ├── group_2_vowel_distinction/")
    print(f"       └── ...")
    
    print(f"\n🎧 Listen & Compare:")
    print(f"   - If sounds IDENTICAL → LoRA not loaded (bug)")
    print(f"   - If sounds DIFFERENT → LoRA working! 🎉")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
