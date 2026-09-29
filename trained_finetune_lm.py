from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa

from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "microsoft/wavlm-base-plus"

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/validation.csv"
TEST_CSV = "data/processed/test.csv"

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    MODEL_DIR / "cremad_wavlm_bigru_attention.pt"
)

SAMPLE_RATE = 16000
MAX_SECONDS = 5
MAX_LENGTH = SAMPLE_RATE * MAX_SECONDS

NUM_CLASSES = 6

EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
}

EMOTION_TO_ID = {
    value: key
    for key, value in EMOTIONS.items()
}


# ============================================================
# TRAINING CONFIG
# ============================================================

BATCH_SIZE = 2

# Effective batch size = 2 x 4 = 8
GRAD_ACCUMULATION = 4

EPOCHS = 12
PATIENCE = 3

WAVLM_LR = 1e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-4

UNFREEZE_LAST_N = 4

NUM_WORKERS = 0

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("high")


# ============================================================
# PRINT SYSTEM INFO
# ============================================================

print("=" * 75)
print("CREMA-D EXPERIMENT #2")
print("WavLM + Partial Fine-Tuning + BiGRU + Attention")
print("=" * 75)

print(f"Device       : {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM         : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(f"Batch size   : {BATCH_SIZE}")
print(f"Accumulation : {GRAD_ACCUMULATION}")
print(f"Effective BS : {BATCH_SIZE * GRAD_ACCUMULATION}")
print(f"Epochs       : {EPOCHS}")
print(f"Unfrozen WavLM layers: {UNFREEZE_LAST_N}")

print()


# ============================================================
# DATASET
# ============================================================

class CREMADDataset(Dataset):

    def __init__(
        self,
        csv_file,
        augment=False,
    ):

        self.df = pd.read_csv(csv_file)
        self.augment = augment

        

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):

        audio, _ = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True,
        )

        return torch.tensor(
            audio,
            dtype=torch.float32
        )

    def _fix_length(self, audio):

        original_length = len(audio)

        # ----------------------------------------------------
        # CROP
        # ----------------------------------------------------

        if original_length > MAX_LENGTH:

            if self.augment:

                start = random.randint(
                    0,
                    original_length - MAX_LENGTH
                )

            else:

                start = (
                    original_length - MAX_LENGTH
                ) // 2

            audio = audio[
                start:start + MAX_LENGTH
            ]

            valid_length = MAX_LENGTH

        # ----------------------------------------------------
        # PAD
        # ----------------------------------------------------

        else:

            valid_length = original_length

            padding = (
                MAX_LENGTH - original_length
            )

            audio = torch.nn.functional.pad(
                audio,
                (0, padding)
            )

        return audio, valid_length

    def _normalize(self, audio, valid_length):

        valid_audio = audio[
            :valid_length
        ]

        mean = valid_audio.mean()

        std = valid_audio.std()

        audio = audio.clone()

        audio[
            :valid_length
        ] = (
            valid_audio - mean
        ) / (
            std + 1e-7
        )

        # Keep padding at exactly zero
        if valid_length < MAX_LENGTH:
            audio[
                valid_length:
            ] = 0

        return audio

    def _augment_audio(
        self,
        audio,
        valid_length,
    ):

        audio = audio.clone()

        valid = audio[
            :valid_length
        ]

        # ----------------------------------------------------
        # RANDOM GAIN
        # ----------------------------------------------------

        if random.random() < 0.5:

            gain = random.uniform(
                0.80,
                1.20
            )

            valid = valid * gain

        # ----------------------------------------------------
        # GAUSSIAN NOISE
        # ----------------------------------------------------

        if random.random() < 0.30:

            noise_scale = random.uniform(
                0.001,
                0.004
            )

            noise = torch.randn_like(
                valid
            ) * noise_scale

            valid = valid + noise

        # ----------------------------------------------------
        # TIME SHIFT
        # ----------------------------------------------------

        if random.random() < 0.30:

            shift = random.randint(
                -1600,
                1600
            )

            if shift > 0:

                shifted = torch.zeros_like(
                    valid
                )

                shifted[shift:] = valid[
                    :-shift
                ]

                valid = shifted

            elif shift < 0:

                shifted = torch.zeros_like(
                    valid
                )

                shifted[:shift] = valid[
                    -shift:
                ]

                valid = shifted

        # ----------------------------------------------------
        # REPLACE VALID AUDIO
        # ----------------------------------------------------

        audio[
            :valid_length
        ] = valid

        # Ensure padding stays zero
        if valid_length < MAX_LENGTH:
            audio[
                valid_length:
            ] = 0

        audio = torch.clamp(
            audio,
            -5.0,
            5.0
        )

        return audio

    def __getitem__(self, index):

        row = self.df.iloc[index]

        audio_path = row["path"]

        audio = self._load_audio(
            audio_path
        )

        audio, valid_length = (
            self._fix_length(audio)
        )

        audio = self._normalize(
            audio,
            valid_length
        )

        if self.augment:

            audio = self._augment_audio(
                audio,
                valid_length
            )

        # ----------------------------------------------------
        # RAW AUDIO ATTENTION MASK
        # ----------------------------------------------------

        attention_mask = torch.zeros(
            MAX_LENGTH,
            dtype=torch.long
        )

        attention_mask[
            :valid_length
        ] = 1

        label = EMOTION_TO_ID[
            row["emotion"]
        ]

        return {
            "audio": audio,
            "attention_mask": attention_mask,
            "label": torch.tensor(
                label,
                dtype=torch.long
            ),
        }


# ============================================================
# DATASETS
# ============================================================

print("Loading datasets...")

train_dataset = CREMADDataset(
    TRAIN_CSV,
    augment=True
)

val_dataset = CREMADDataset(
    VAL_CSV,
    augment=False
)

test_dataset = CREMADDataset(
    TEST_CSV,
    augment=False
)

print(
    f"Train: {len(train_dataset)}"
)

print(
    f"Validation: {len(val_dataset)}"
)

print(
    f"Test: {len(test_dataset)}"
)

print()


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
)


# ============================================================
# MODEL
# ============================================================

class WavLMBiGRUAttention(nn.Module):

    def __init__(self):

        super().__init__()

        print("Loading WavLM...")

        self.wavlm = WavLMModel.from_pretrained(
            MODEL_NAME
        )

        wavlm_hidden = (
            self.wavlm.config.hidden_size
        )

        # ----------------------------------------------------
        # FREEZE EVERYTHING
        # ----------------------------------------------------

        for parameter in self.wavlm.parameters():
            parameter.requires_grad = False

        # ----------------------------------------------------
        # UNFREEZE TOP WAVLM LAYERS
        # ----------------------------------------------------

        layers = self.wavlm.encoder.layers

        if UNFREEZE_LAST_N > len(layers):

            raise ValueError(
                "UNFREEZE_LAST_N is larger than "
                "number of WavLM encoder layers."
            )

        for layer in layers[
            -UNFREEZE_LAST_N:
        ]:

            for parameter in layer.parameters():

                parameter.requires_grad = True

        # Unfreeze final encoder layer norm
        if hasattr(
            self.wavlm.encoder,
            "layer_norm"
        ):

            for parameter in (
                self.wavlm.encoder
                .layer_norm
                .parameters()
            ):

                parameter.requires_grad = True

        # ----------------------------------------------------
        # BiGRU
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=wavlm_hidden,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )

        gru_dim = 512

        # ----------------------------------------------------
        # MULTI-HEAD ATTENTION
        # ----------------------------------------------------

        self.mha = nn.MultiheadAttention(
            embed_dim=gru_dim,
            num_heads=8,
            dropout=0.15,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(
            gru_dim
        )

        # ----------------------------------------------------
        # FEED FORWARD NETWORK
        # ----------------------------------------------------

        self.ffn = nn.Sequential(
            nn.Linear(
                gru_dim,
                1024
            ),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(
                1024,
                gru_dim
            ),
        )

        self.ffn_norm = nn.LayerNorm(
            gru_dim
        )

        # ----------------------------------------------------
        # ATTENTIVE POOLING
        # ----------------------------------------------------

        self.pool_projection = nn.Linear(
            gru_dim,
            128
        )

        self.pool_score = nn.Linear(
            128,
            1
        )

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        # Mean + weighted std
        pooled_dim = gru_dim * 2

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                pooled_dim
            ),

            nn.Linear(
                pooled_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.35),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(
                128,
                NUM_CLASSES
            ),
        )

    def forward(
        self,
        input_values,
        attention_mask,
    ):

        # ----------------------------------------------------
        # WAVLM
        # ----------------------------------------------------

        wavlm_output = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        x = wavlm_output.last_hidden_state

        # [B, T, 768]

        # ----------------------------------------------------
        # GET FEATURE-LEVEL MASK
        # ----------------------------------------------------

        try:

            feature_mask = (
                self.wavlm
                ._get_feature_vector_attention_mask(
                    x.shape[1],
                    attention_mask,
                )
            )

        except AttributeError:

            # Fallback based on observed sequence length
            lengths = attention_mask.sum(
                dim=1
            )

            feature_lengths = torch.ceil(
                lengths.float()
                / MAX_LENGTH
                * x.shape[1]
            ).long()

            feature_mask = torch.zeros(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )

            for i, length in enumerate(
                feature_lengths
            ):

                feature_mask[
                    i,
                    :length
                ] = True

        # ----------------------------------------------------
        # ZERO PADDED FEATURES
        # ----------------------------------------------------

        x = x.masked_fill(
            ~feature_mask.unsqueeze(-1),
            0.0
        )

        # ----------------------------------------------------
        # BiGRU
        # ----------------------------------------------------

        x, _ = self.gru(
            x
        )

        # ----------------------------------------------------
        # MULTI-HEAD ATTENTION
        # ----------------------------------------------------

        key_padding_mask = ~feature_mask

        attention_output, _ = (
            self.mha(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        )

        x = self.attn_norm(
            x + attention_output
        )

        # ----------------------------------------------------
        # FEED FORWARD
        # ----------------------------------------------------

        x = self.ffn_norm(
            x + self.ffn(x)
        )

        # ----------------------------------------------------
        # ATTENTIVE STATISTICS POOLING
        # ----------------------------------------------------

        score_features = torch.tanh(
            self.pool_projection(x)
        )

        scores = self.pool_score(
            score_features
        ).squeeze(-1)

        scores = scores.masked_fill(
            ~feature_mask,
            -1e4
        )

        weights = torch.softmax(
            scores,
            dim=1
        ).unsqueeze(-1)

        # Weighted mean
        mean = torch.sum(
            weights * x,
            dim=1
        )

        # Weighted standard deviation
        variance = torch.sum(
            weights *
            (x - mean.unsqueeze(1)).pow(2),
            dim=1
        )

        std = torch.sqrt(
            variance + 1e-5
        )

        pooled = torch.cat(
            [mean, std],
            dim=1
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        logits = self.classifier(
            pooled
        )

        return logits


# ============================================================
# CREATE MODEL
# ============================================================

model = WavLMBiGRUAttention()

model = model.to(
    DEVICE
)


# ============================================================
# PARAMETER COUNTS
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

wavlm_trainable = sum(
    p.numel()
    for p in model.wavlm.parameters()
    if p.requires_grad
)

head_trainable = (
    trainable_params
    - wavlm_trainable
)

print()
print("=" * 75)
print("MODEL PARAMETERS")
print("=" * 75)

print(
    f"Total parameters     : "
    f"{total_params:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_params:,}"
)

print(
    f"WavLM trainable      : "
    f"{wavlm_trainable:,}"
)

print(
    f"Head trainable       : "
    f"{head_trainable:,}"
)

print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# ============================================================
# PARAMETER GROUPS
# ============================================================

wavlm_parameters = []
head_parameters = []

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    if name.startswith("wavlm."):

        wavlm_parameters.append(
            parameter
        )

    else:

        head_parameters.append(
            parameter
        )


optimizer = torch.optim.AdamW(
    [
        {
            "params": wavlm_parameters,
            "lr": WAVLM_LR,
        },

        {
            "params": head_parameters,
            "lr": HEAD_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
    min_lr=1e-7,
)


# ============================================================
# MIXED PRECISION
# ============================================================

if USE_AMP:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_items = 0

    predictions = []
    targets = []

    optimizer.zero_grad(
        set_to_none=True
    )

    start_time = time.time()

    for batch_idx, batch in enumerate(
        train_loader
    ):

        audio = batch[
            "audio"
        ].to(
            DEVICE,
            non_blocking=True
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch[
            "label"
        ].to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        if USE_AMP:

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
            ):

                logits = model(
                    audio,
                    attention_mask,
                )

                loss = criterion(
                    logits,
                    labels,
                )

                loss_for_backward = (
                    loss
                    / GRAD_ACCUMULATION
                )

        else:

            logits = model(
                audio,
                attention_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss_for_backward = (
                loss
                / GRAD_ACCUMULATION
            )

        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        if scaler is not None:

            scaler.scale(
                loss_for_backward
            ).backward()

        else:

            loss_for_backward.backward()

        # ----------------------------------------------------
        # OPTIMIZER STEP
        # ----------------------------------------------------

        if (
            (batch_idx + 1)
            % GRAD_ACCUMULATION
            == 0
            or
            (batch_idx + 1)
            == len(train_loader)
        ):

            if scaler is not None:

                scaler.unscale_(
                    optimizer
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            if scaler is not None:

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_items += batch_size

        preds = logits.argmax(
            dim=1
        )

        predictions.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            batch_idx + 1
        ) % 100 == 0:

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"  Batch "
                f"{batch_idx + 1}/"
                f"{len(train_loader)} "
                f"| Loss {loss.item():.4f} "
                f"| Time {elapsed / 60:.1f}m"
            )

    average_loss = (
        total_loss
        / total_items
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    total_loss = 0.0
    total_items = 0

    predictions = []
    targets = []

    for batch in loader:

        audio = batch[
            "audio"
        ].to(
            DEVICE,
            non_blocking=True
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch[
            "label"
        ].to(
            DEVICE,
            non_blocking=True
        )

        if USE_AMP:

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
            ):

                logits = model(
                    audio,
                    attention_mask,
                )

                loss = criterion(
                    logits,
                    labels
                )

        else:

            logits = model(
                audio,
                attention_mask,
            )

            loss = criterion(
                logits,
                labels
            )

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_items += batch_size

        preds = logits.argmax(
            dim=1
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        targets.extend(
            labels.cpu().numpy()
        )

    average_loss = (
        total_loss
        / total_items
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        predictions,
        targets,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

print()
print("=" * 75)
print("TRAINING")
print("=" * 75)

best_val_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(
    1,
    EPOCHS + 1
):

    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print("-" * 75)

    train_loss, train_acc, train_f1 = (
        train_one_epoch()
    )

    val_loss, val_acc, val_f1, _, _ = (
        evaluate(
            val_loader
        )
    )

    scheduler.step(
        val_f1
    )

    current_wavlm_lr = (
        optimizer.param_groups[0]["lr"]
    )

    current_head_lr = (
        optimizer.param_groups[1]["lr"]
    )

    print()

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_acc * 100:.2f}%"
    )

    print(
        f"Train F1   : "
        f"{train_f1:.4f}"
    )

    print()

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_acc * 100:.2f}%"
    )

    print(
        f"Val F1     : "
        f"{val_f1:.4f}"
    )

    print()

    print(
        f"WavLM LR   : "
        f"{current_wavlm_lr:.2e}"
    )

    print(
        f"Head LR    : "
        f"{current_head_lr:.2e}"
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "best_epoch": epoch,
                "model_name": MODEL_NAME,
                "unfreeze_last_n": UNFREEZE_LAST_N,
                "emotion_to_id": EMOTION_TO_ID,
            },
            BEST_MODEL_PATH,
        )

        print()
        print(
            f"✓ BEST MODEL SAVED"
            f" | Val F1 = {val_f1:.4f}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 75)
print("LOADING BEST MODEL")
print("=" * 75)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

print(
    f"Best epoch : "
    f"{checkpoint['best_epoch']}"
)

print(
    f"Best Val F1: "
    f"{checkpoint['val_macro_f1']:.4f}"
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 75)
print("FINAL TEST")
print("=" * 75)

test_loss, test_acc, test_f1, predictions, targets = evaluate(
    test_loader
)

print()

print(
    f"Test Accuracy : "
    f"{test_acc * 100:.2f}%"
)

print(
    f"Test Macro F1 : "
    f"{test_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    targets,
    predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=[
        EMOTIONS[i]
        for i in range(NUM_CLASSES)
    ],
    digits=4,
    zero_division=0,
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    targets,
    predictions,
    labels=list(range(NUM_CLASSES))
)

print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print()

print(
    f"{'':12s}"
    +
    "".join(
        f"{EMOTIONS[i][:10]:>12s}"
        for i in range(NUM_CLASSES)
    )
)

for i in range(NUM_CLASSES):

    row = "".join(
        f"{cm[i, j]:>12d}"
        for j in range(NUM_CLASSES)
    )

    print(
        f"{EMOTIONS[i][:10]:>10s}  "
        f"{row}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    RESULTS_DIR
    / "cremad_wavlm_bigru_attention_results.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CREMA-D EXPERIMENT #2\n"
    )

    f.write(
        "WavLM + Partial Fine-Tuning + "
        "BiGRU + Attention\n\n"
    )

    f.write(
        f"Best epoch: {best_epoch}\n"
    )

    f.write(
        f"Best validation F1: "
        f"{best_val_f1:.4f}\n"
    )

    f.write(
        f"Test accuracy: "
        f"{test_acc:.4f}\n"
    )

    f.write(
        f"Test Macro F1: "
        f"{test_f1:.4f}\n\n"
    )

    f.write(
        "Classification report:\n"
    )

    f.write(report)

    f.write(
        "\nConfusion matrix:\n"
    )

    f.write(
        np.array2string(cm)
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("EXPERIMENT #2 COMPLETE")
print("=" * 75)

print(
    f"Best validation F1 : "
    f"{best_val_f1:.4f}"
)

print(
    f"Test accuracy       : "
    f"{test_acc * 100:.2f}%"
)

print(
    f"Test Macro F1       : "
    f"{test_f1:.4f}"
)

print()
print(
    f"Model saved: "
    f"{BEST_MODEL_PATH}"
)

print(
    f"Results saved: "
    f"{results_file}"
)