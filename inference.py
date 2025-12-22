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


def main():
    parser = argparse.ArgumentParser(description="Viterbox Text-to-Speech")
    parser.add_argument("--model_path", "-m", type=str, default=None, help="Path to local model checkpoint (optional)")
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
        if args.model_path:
            print(f"Loading from local path: {args.model_path}")
            tts = Viterbox.from_local(args.model_path, device=args.device)
        else:
            print("Loading from HuggingFace Hub...")
            tts = Viterbox.from_pretrained(args.device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
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
