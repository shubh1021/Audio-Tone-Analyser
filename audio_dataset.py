from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000
MAX_SECONDS = 5
MAX_LENGTH = SAMPLE_RATE * MAX_SECONDS

EMOTION_TO_ID = {
    "angry": 0,
    "disgust": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
}

ID_TO_EMOTION = {
    value: key
    for key, value in EMOTION_TO_ID.items()
}


# ============================================================
# DATASET
# ============================================================

class CREMADAudioDataset(Dataset):

    def __init__(
        self,
        csv_file,
        augment=False,
    ):

        self.csv_file = Path(csv_file)
        self.augment = augment

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"CSV file not found:\n"
                f"{self.csv_file.resolve()}"
            )

        self.df = pd.read_csv(self.csv_file)

        required = {
            "path",
            "emotion",
        }

        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):

        audio, _ = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True,
        )

        audio = audio.astype(np.float32)

        # ----------------------------------------------------
        # FIX LENGTH
        # ----------------------------------------------------

        if len(audio) > MAX_LENGTH:

            if self.augment:
                # Random crop during training
                start = np.random.randint(
                    0,
                    len(audio) - MAX_LENGTH + 1
                )
            else:
                # Center crop during validation/test
                start = (
                    len(audio) - MAX_LENGTH
                ) // 2

            audio = audio[
                start:start + MAX_LENGTH
            ]

        else:

            pad_amount = MAX_LENGTH - len(audio)

            audio = np.pad(
                audio,
                (0, pad_amount),
                mode="constant",
            )

        return audio

    def _augment_audio(self, audio):

        # ----------------------------------------------------
        # RANDOM GAIN
        # ----------------------------------------------------

        if np.random.rand() < 0.5:

            gain = np.random.uniform(
                0.8,
                1.2
            )

            audio = audio * gain

        # ----------------------------------------------------
        # GAUSSIAN NOISE
        # ----------------------------------------------------

        if np.random.rand() < 0.3:

            noise_level = np.random.uniform(
                0.001,
                0.005
            )

            noise = np.random.randn(
                len(audio)
            ).astype(np.float32)

            audio = audio + noise_level * noise

        # ----------------------------------------------------
        # TIME SHIFT
        # ----------------------------------------------------

        if np.random.rand() < 0.3:

            shift = np.random.randint(
                -1600,
                1601
            )

            audio = np.roll(
                audio,
                shift
            )

            if shift > 0:
                audio[:shift] = 0

            elif shift < 0:
                audio[shift:] = 0

        # ----------------------------------------------------
        # CLIP
        # ----------------------------------------------------

        audio = np.clip(
            audio,
            -1.0,
            1.0
        )

        return audio

    def __getitem__(self, index):

        row = self.df.iloc[index]

        audio_path = Path(row["path"])

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found:\n"
                f"{audio_path.resolve()}"
            )

        audio = self._load_audio(
            audio_path
        )

        if self.augment:
            audio = self._augment_audio(
                audio
            )

        label = EMOTION_TO_ID[
            row["emotion"]
        ]

        return {
            "audio": torch.tensor(
                audio,
                dtype=torch.float32
            ),
            "label": torch.tensor(
                label,
                dtype=torch.long
            ),
        }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    dataset = CREMADAudioDataset(
        "data/processed/train.csv",
        augment=True,
    )

    print("=" * 70)
    print("CREMA-D AUDIO DATASET TEST")
    print("=" * 70)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    print(
        f"Audio shape : "
        f"{sample['audio'].shape}"
    )

    print(
        f"Audio dtype : "
        f"{sample['audio'].dtype}"
    )

    print(
        f"Label       : "
        f"{sample['label'].item()}"
    )

    print(
        f"Emotion     : "
        f"{ID_TO_EMOTION[sample['label'].item()]}"
    )

    print(
        f"Audio min   : "
        f"{sample['audio'].min().item():.4f}"
    )

    print(
        f"Audio max   : "
        f"{sample['audio'].max().item():.4f}"
    )

    print()
    print("✓ Dataset loader working.")