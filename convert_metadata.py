"""
Convert raw metadata to voice-specific CSV files for preprocessing.

Input format:
    audio_path,voice_id,text

Output format (per voice_id):
    audio|transcript
"""

import pandas as pd
from pathlib import Path
import argparse


def convert_metadata(
    input_file: str, 
    output_dir: str = "./metadata_by_voice",
    old_prefix: str = None,
    new_prefix: str = None,
):
    """
    Convert raw metadata to voice-specific CSV files.
    
    Args:
        input_file: Path to input CSV with columns: audio_path, voice_id, text
        output_dir: Directory to save voice-specific CSVs
        old_prefix: Old path prefix to replace (e.g., "/kaggle/working/output/output/")
        new_prefix: New path prefix (e.g., "/kaggle/input/sub-dolly/output/output")
    """
    # Read input metadata
    print(f"📖 Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Validate columns
    required_cols = {'audio_path', 'voice_id', 'text'}
    if not required_cols.issubset(df.columns):
        print(f"❌ Missing columns! Required: {required_cols}, Found: {set(df.columns)}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Replace path prefix if specified
    if old_prefix and new_prefix:
        print(f"\n🔄 Replacing paths:")
        print(f"   Old: {old_prefix}")
        print(f"   New: {new_prefix}")
        df['audio_path'] = df['audio_path'].str.replace(old_prefix, new_prefix, regex=False)
        replaced_count = (df['audio_path'].str.contains(new_prefix)).sum()
        print(f"   ✅ Replaced {replaced_count} paths")
    
    # Group by voice_id
    voice_groups = df.groupby('voice_id')
    
    print(f"\n📊 Found {len(voice_groups)} unique voices:")
    for voice_id, group in voice_groups:
        print(f"  - {voice_id}: {len(group)} samples")
    
    print(f"\n💾 Saving individual CSV files to {output_dir}/...")
    
    # Save each voice to separate CSV
    for voice_id, group in voice_groups:
        # Create output dataframe with required format
        output_df = pd.DataFrame({
            'audio': group['audio_path'],
            'transcript': group['text']
        })
        
        # Clean voice_id for filename (remove spaces, special chars)
        safe_voice_id = voice_id.replace(' ', '_').replace('/', '_')
        output_file = output_path / f"{safe_voice_id}.csv"
        
        # Save with pipe separator
        output_df.to_csv(output_file, sep='|', index=False)
        print(f"  ✅ {output_file} ({len(output_df)} samples)")
    
    # Also create a combined file for all voices
    combined_df = pd.DataFrame({
        'audio': df['audio_path'],
        'transcript': df['text']
    })
    combined_file = output_path / "metadata_all.csv"
    combined_df.to_csv(combined_file, sep='|', index=False)
    print(f"\n  ✅ {combined_file} (ALL {len(combined_df)} samples)")
    
    print(f"\n🎉 Done! Created {len(voice_groups)} voice-specific files + 1 combined file")


def main():
    parser = argparse.ArgumentParser(description="Convert metadata to voice-specific CSVs")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output_dir", type=str, default="./metadata_by_voice", 
                        help="Output directory for voice-specific CSVs")
    parser.add_argument("--old_prefix", type=str, default=None,
                        help="Old path prefix to replace (e.g., '/kaggle/working/output/output/')")
    parser.add_argument("--new_prefix", type=str, default=None,
                        help="New path prefix (e.g., '/kaggle/input/sub-dolly/output/output')")
    
    args = parser.parse_args()
    convert_metadata(args.input, args.output_dir, args.old_prefix, args.new_prefix)


if __name__ == "__main__":
    # Example usage:
    # python convert_metadata.py --input raw_metadata.csv --output_dir ./metadata_by_voice
    main()
