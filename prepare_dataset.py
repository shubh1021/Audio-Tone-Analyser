from pathlib import Path
import pandas as pd
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

AUDIO_DIR = Path("archive/AudioWAV")
OUTPUT_FILE = Path("data/cremad_metadata.csv")

EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


# ============================================================
# CHECK DATASET
# ============================================================

if not AUDIO_DIR.exists():
    raise FileNotFoundError(
        f"Audio directory not found:\n{AUDIO_DIR.resolve()}"
    )

wav_files = sorted(AUDIO_DIR.glob("*.wav"))

print("=" * 70)
print("CREMA-D DATASET CHECK")
print("=" * 70)

print(f"Audio directory : {AUDIO_DIR.resolve()}")
print(f"WAV files found : {len(wav_files)}")


# ============================================================
# PARSE FILENAMES
# ============================================================

records = []

for wav_path in wav_files:

    # Example:
    # 1001_IEO_ANG_HI.wav

    parts = wav_path.stem.split("_")

    if len(parts) != 4:
        print(f"Skipping malformed filename: {wav_path.name}")
        continue

    actor_id, sentence, emotion_code, level = parts

    if emotion_code not in EMOTIONS:
        print(
            f"Skipping unknown emotion code "
            f"{emotion_code}: {wav_path.name}"
        )
        continue

    records.append({
        "path": str(wav_path.as_posix()),
        "filename": wav_path.name,
        "actor_id": int(actor_id),
        "sentence": sentence,
        "emotion": EMOTIONS[emotion_code],
        "emotion_code": emotion_code,
        "level": level,
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("No valid CREMA-D audio files were found.")


# ============================================================
# DATASET STATISTICS
# ============================================================

print()
print("-" * 70)
print("VALID DATA")
print("-" * 70)

print(f"Valid clips       : {len(df)}")
print(f"Unique actors     : {df['actor_id'].nunique()}")
print(f"Unique sentences  : {df['sentence'].nunique()}")
print(f"Unique emotions   : {df['emotion'].nunique()}")

print()
print("Emotion distribution:")
print(df["emotion"].value_counts().sort_index())

print()
print("Actor count:")
print(df["actor_id"].nunique())

print()
print("Intensity/level distribution:")
print(df["level"].value_counts().sort_index())


# ============================================================
# VERIFY SIX EMOTIONS
# ============================================================

expected_emotions = {
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
}

found_emotions = set(df["emotion"].unique())

if found_emotions != expected_emotions:

    missing = expected_emotions - found_emotions
    extra = found_emotions - expected_emotions

    print()
    print("WARNING: Emotion set mismatch")

    if missing:
        print("Missing:", sorted(missing))

    if extra:
        print("Unexpected:", sorted(extra))

else:
    print()
    print("✓ All 6 emotions detected correctly.")


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicates = df["filename"].duplicated().sum()

print()
print(f"Duplicate filenames: {duplicates}")

if duplicates != 0:
    raise RuntimeError("Duplicate filenames detected.")


# ============================================================
# SAVE METADATA
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print()
print("-" * 70)
print(f"Metadata saved to: {OUTPUT_FILE}")
print("-" * 70)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(f"Clips   : {len(df)}")
print(f"Actors  : {df['actor_id'].nunique()}")
print(f"Classes : {df['emotion'].nunique()}")

for emotion, count in sorted(
    df["emotion"].value_counts().items()
):
    print(f"{emotion:10s}: {count}")


print()
print("Example records:")
print(df.head(10).to_string(index=False))