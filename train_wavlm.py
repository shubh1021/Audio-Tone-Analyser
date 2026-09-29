from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import WavLMModel

from audio_dataset import (
    CREMADAudioDataset,
    EMOTION_TO_ID,
    ID_TO_EMOTION,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "microsoft/wavlm-base-plus"

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/validation.csv"
TEST_CSV = "data/processed/test.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "wavlm_baseline.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 6

NUM_WORKERS = 0       # Windows-safe
PIN_MEMORY = DEVICE.type == "cuda"

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE INFO
# ============================================================

print("=" * 70)
print("CREMA-D WavLM BASELINE")
print("=" * 70)

print(f"Device      : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory  : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(f"Model       : {MODEL_NAME}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Epochs      : {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

print()


# ============================================================
# DATASETS
# ============================================================

print("Loading datasets...")

train_dataset = CREMADAudioDataset(
    TRAIN_CSV,
    augment=False,
)

val_dataset = CREMADAudioDataset(
    VAL_CSV,
    augment=False,
)

test_dataset = CREMADAudioDataset(
    TEST_CSV,
    augment=False,
)

print(f"Train clips : {len(train_dataset)}")
print(f"Val clips   : {len(val_dataset)}")
print(f"Test clips  : {len(test_dataset)}")

print()


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


# ============================================================
# MODEL
# ============================================================

class WavLMEmotionClassifier(nn.Module):

    def __init__(self, model_name, num_classes):

        super().__init__()

        print("Loading WavLM...")

        self.wavlm = WavLMModel.from_pretrained(
            model_name
        )

        hidden_size = self.wavlm.config.hidden_size

        print(
            f"WavLM hidden size: {hidden_size}"
        )

        # ----------------------------------------------------
        # FREEZE WAVLM
        # ----------------------------------------------------

        for param in self.wavlm.parameters():
            param.requires_grad = False

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.dropout = nn.Dropout(0.30)

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, input_values):

        # ----------------------------------------------------
        # WAVLM
        # ----------------------------------------------------

        outputs = self.wavlm(
            input_values=input_values
        )

        hidden_states = outputs.last_hidden_state

        # Shape:
        # [batch, time, hidden]

        # ----------------------------------------------------
        # MEAN POOLING
        # ----------------------------------------------------

        pooled = hidden_states.mean(
            dim=1
        )

        # Shape:
        # [batch, hidden]

        pooled = self.dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        return logits


model = WavLMEmotionClassifier(
    MODEL_NAME,
    NUM_CLASSES,
)

model = model.to(DEVICE)


# ============================================================
# COUNT TRAINABLE PARAMETERS
# ============================================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print()
print("=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)

print(
    f"Total parameters     : "
    f"{total_params:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_params:,}"
)

print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# MIXED PRECISION
# ============================================================

USE_AMP = DEVICE.type == "cuda"

scaler = torch.cuda.amp.GradScaler(
    enabled=USE_AMP
)


# ============================================================
# NORMALIZE AUDIO
# ============================================================

def normalize_audio(audio):

    # audio shape:
    # [batch, samples]

    mean = audio.mean(
        dim=1,
        keepdim=True
    )

    std = audio.std(
        dim=1,
        keepdim=True
    )

    audio = (
        audio - mean
    ) / (
        std + 1e-7
    )

    return audio


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    all_predictions,
    all_targets,
):

    predictions = np.array(
        all_predictions
    )

    targets = np.array(
        all_targets
    )

    accuracy = (
        predictions == targets
    ).mean()

    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    f1_scores = []

    for class_id in range(NUM_CLASSES):

        tp = np.sum(
            (predictions == class_id)
            &
            (targets == class_id)
        )

        fp = np.sum(
            (predictions == class_id)
            &
            (targets != class_id)
        )

        fn = np.sum(
            (predictions != class_id)
            &
            (targets == class_id)
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        f1_scores.append(f1)

    macro_f1 = np.mean(
        f1_scores
    )

    return accuracy, macro_f1


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    all_predictions = []
    all_targets = []

    for batch_idx, batch in enumerate(
        train_loader
    ):

        audio = batch["audio"].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch["label"].to(
            DEVICE,
            non_blocking=True
        )

        audio = normalize_audio(
            audio
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):

            logits = model(
                audio
            )

            loss = criterion(
                logits,
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += (
            loss.item()
            * audio.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        if (
            batch_idx + 1
        ) % 100 == 0:

            print(
                f"  Batch "
                f"{batch_idx + 1}/"
                f"{len(train_loader)} "
                f"| Loss: "
                f"{loss.item():.4f}"
            )

    epoch_loss = (
        total_loss
        / len(train_dataset)
    )

    accuracy, macro_f1 = calculate_metrics(
        all_predictions,
        all_targets
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_loader(loader, dataset):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_targets = []

    for batch in loader:

        audio = batch["audio"].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch["label"].to(
            DEVICE,
            non_blocking=True
        )

        audio = normalize_audio(
            audio
        )

        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):

            logits = model(
                audio
            )

            loss = criterion(
                logits,
                labels
            )

        total_loss += (
            loss.item()
            * audio.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

    epoch_loss = (
        total_loss
        / len(dataset)
    )

    accuracy, macro_f1 = calculate_metrics(
        all_predictions,
        all_targets
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        all_predictions,
        all_targets,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_f1 = -1.0
best_epoch = 0

print("=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(
    1,
    EPOCHS + 1
):

    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )
    print("-" * 70)

    train_loss, train_acc, train_f1 = (
        train_one_epoch()
    )

    val_loss, val_acc, val_f1, _, _ = (
        evaluate_loader(
            val_loader,
            val_dataset
        )
    )

    print()
    print(
        f"Train Loss : {train_loss:.4f}"
    )
    print(
        f"Train Acc  : {train_acc:.4f}"
        f" ({train_acc * 100:.2f}%)"
    )
    print(
        f"Train F1   : {train_f1:.4f}"
    )

    print()

    print(
        f"Val Loss   : {val_loss:.4f}"
    )
    print(
        f"Val Acc    : {val_acc:.4f}"
        f" ({val_acc * 100:.2f}%)"
    )
    print(
        f"Val F1     : {val_f1:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_accuracy": val_acc,
                "model_name": MODEL_NAME,
                "emotion_to_id": EMOTION_TO_ID,
            },
            BEST_MODEL_PATH,
        )

        print()
        print(
            f"✓ Best model saved "
            f"(Val F1: {val_f1:.4f})"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch : {checkpoint['epoch']}"
)

print(
    f"Best Val F1: "
    f"{checkpoint['val_f1']:.4f}"
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 70)
print("FINAL TEST")
print("=" * 70)

test_loss, test_acc, test_f1, predictions, targets = (
    evaluate_loader(
        test_loader,
        test_dataset
    )
)

print()
print(
    f"Test Loss : {test_loss:.4f}"
)

print(
    f"Accuracy  : "
    f"{test_acc:.4f} "
    f"({test_acc * 100:.2f}%)"
)

print(
    f"Macro F1  : "
    f"{test_f1:.4f}"
)


# ============================================================
# CLASS-BY-CLASS RESULTS
# ============================================================

print()
print("=" * 70)
print("CLASS RESULTS")
print("=" * 70)

for class_id in range(
    NUM_CLASSES
):

    class_name = ID_TO_EMOTION[
        class_id
    ]

    class_predictions = np.array(
        predictions
    ) == class_id

    class_targets = np.array(
        targets
    ) == class_id

    tp = np.sum(
        class_predictions
        &
        class_targets
    )

    fp = np.sum(
        class_predictions
        &
        ~class_targets
    )

    fn = np.sum(
        ~class_predictions
        &
        class_targets
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    support = np.sum(
        class_targets
    )

    print(
        f"{class_name:10s} "
        f"| Precision: {precision:.3f} "
        f"| Recall: {recall:.3f} "
        f"| F1: {f1:.3f} "
        f"| Support: {support}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=int
)

for target, prediction in zip(
    targets,
    predictions
):

    confusion[
        target,
        prediction
    ] += 1


header = [
    ID_TO_EMOTION[i]
    for i in range(NUM_CLASSES)
]

print()
print(
    " " * 12
    +
    "".join(
        f"{name[:9]:>11}"
        for name in header
    )
)

for i in range(NUM_CLASSES):

    row_name = ID_TO_EMOTION[i]

    row = "".join(
        f"{confusion[i, j]:>11}"
        for j in range(NUM_CLASSES)
    )

    print(
        f"{row_name[:10]:>10}  "
        f"{row}"
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation F1 : "
    f"{checkpoint['val_f1']:.4f}"
)

print(
    f"Final test accuracy: "
    f"{test_acc:.4f}"
)

print(
    f"Final test Macro F1 : "
    f"{test_f1:.4f}"
)

print()
print(
    f"Saved model: "
    f"{BEST_MODEL_PATH}"
)