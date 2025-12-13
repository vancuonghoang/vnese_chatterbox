import os
import subprocess
import pandas as pd

# ======================
# CONFIG
# ======================
BASE_DIR = "/Users/cuonghoang1611/Desktop/dolly"
PARQUET_DIR = f"{BASE_DIR}/parquet"
OUTPUT_DIR = f"{BASE_DIR}/output"
CSV_PATH = f"{OUTPUT_DIR}/metadata.csv"

BASE_URL = "https://huggingface.co/datasets/dolly-vn/dolly-audio-1000h-vietnamese/resolve/main/data"

START_IDX = 0
END_IDX = 15  # xử lý train-00000 → train-00015

TARGET_VOICES = {
    "Confident Woman",
    "Reliable Man",
    "Bossy Leader",
    "Calm Woman",
    "Wise Scholar",
    "Wise lady",
    "Strong-willed Boy",
    "Gentle Woman",
    "Deep-voiced Gentleman",
    "Confident Woman",
    "Reliable Man"
}

# ======================
# SETUP
# ======================
os.makedirs(PARQUET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_name(name):
    return name.strip().replace(" ", "_")

# ======================
# PROCESS LOOP
# ======================
for i in range(START_IDX, END_IDX + 1):
    fname = f"train-{i:05d}-of-00332.parquet"
    parquet_path = os.path.join(PARQUET_DIR, fname)

    # ---- download parquet
    url = f"{BASE_URL}/{fname}?download=true"
    print(f"⬇️ Downloading {fname}")
    subprocess.run(
        ["wget", "-q", url, "-O", parquet_path],
        check=True
    )

    # ---- read parquet
    print(f"📦 Processing {fname}")
    df = pd.read_parquet(parquet_path)

    rows = []

    for _, row in df.iterrows():
        voice_id = row["voice_id"]
        if voice_id not in TARGET_VOICES:
            continue

        voice_dir = os.path.join(OUTPUT_DIR, clean_name(voice_id))
        os.makedirs(voice_dir, exist_ok=True)

        filename = row["audio_filename"]
        audio_bytes = row["audio"]["bytes"]
        text = row["text"]

        audio_path = os.path.join(voice_dir, filename)

        # ---- write wav
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        rows.append({
            "audio_path": audio_path,
            "voice_id": voice_id,
            "text": text
        })

    # ---- append CSV
    if rows:
        pd.DataFrame(rows).to_csv(
            CSV_PATH,
            mode="a",
            index=False,
            header=not os.path.exists(CSV_PATH),
            encoding="utf-8"
        )

    print(f"✅ Extracted {len(rows)} samples")

    # ---- cleanup parquet
    os.remove(parquet_path)
    print(f"🧹 Deleted {fname}\n")

print("🎉 ALL DONE")
