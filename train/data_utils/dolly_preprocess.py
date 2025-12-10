"""
Convert HuggingFace dolly-vn dataset to Viterbox training format.

Output:
- metadata.csv: CSV file with audio|transcript format
- wavs/: Directory containing all audio files
"""

from datasets import load_dataset
import csv
from pathlib import Path
import soundfile as sf

# Configuration
OUTPUT_DIR = Path("./dolly_dataset")
AUDIO_DIR = OUTPUT_DIR / "wavs"
CSV_PATH = OUTPUT_DIR / "metadata.csv"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
print("Loading dataset...")
ds = load_dataset("dolly-vn/dolly-audio-1000h-vietnamese", split="train")

# Convert to Viterbox format
print(f"Converting {len(ds)} samples...")
with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerow(['audio', 'transcript'])  # Header
    
    for idx, sample in enumerate(ds):
        # Extract data
        audio_filename = sample['audio_filename']
        text = sample['text']
        audio_data = sample['audio']
        
        # Save audio to wavs/
        audio_path = AUDIO_DIR / audio_filename
        sf.write(
            audio_path,
            audio_data['array'],
            audio_data['sampling_rate']
        )
        
        # Write to CSV
        writer.writerow([f"wavs/{audio_filename}", text])
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(ds)} samples...")

print(f"\n✅ Conversion complete!")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"📄 Metadata CSV: {CSV_PATH}")
print(f"🎵 Audio files: {AUDIO_DIR} ({len(list(AUDIO_DIR.glob('*.wav')))} files)")

# Next steps
print("\n🔄 Next steps:")
print("1. Run preprocessing:")
print(f"   python preprocess_dataset.py \\")
print(f"       --metadata_csv {CSV_PATH} \\")
print(f"       --audio_dir {AUDIO_DIR} \\")
print(f"       --output_dir ./preprocessed \\")
print(f"       --checkpoint ./vietnamese/pretrained_model_download \\")
print(f"       --device cuda")
print("\n2. Start training:")
print(f"   python viterbox-tts/train/run.py \\")
print(f"       --preprocessed_dir ./preprocessed \\")
print(f"       --output_dir ./checkpoints/dolly_finetune \\")
print(f"       --use_lora --use_wandb")