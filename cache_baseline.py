from pathlib import Path
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import WavLMModel

from audio_dataset import CREMADAudioDataset


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "microsoft/wavlm-base-plus"

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/validation.csv"
TEST_CSV = "data/processed/test.csv"

CACHE_DIR = Path("data/processed/wavlm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Your RTX 4050 should be able to handle this.
# If CUDA out-of-memory occurs, change 8 -> 4.
EMBED_BATCH_SIZE = 8

CLASSIFIER_BATCH_SIZE = 64
EPOCHS = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 6
HIDDEN_SIZE = 768

SEED = 42


# ============================================================
# EMOTION LABELS
# ============================================================

EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("FAST FROZEN-WavLM BASELINE")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print()


# ============================================================
# LOAD WAVLM
# ============================================================

print("Loading WavLM...")

wavlm = WavLMModel.from_pretrained(
    MODEL_NAME
)

wavlm = wavlm.to(DEVICE)
wavlm.eval()

# Absolutely no gradient computation.
for parameter in wavlm.parameters():
    parameter.requires_grad = False

print("✓ WavLM loaded and frozen.")
print()


# ============================================================
# NORMALIZE AUDIO
# ============================================================

def normalize_audio(audio):

    mean = audio.mean(
        dim=1,
        keepdim=True
    )

    std = audio.std(
        dim=1,
        keepdim=True
    )

    return (
        audio - mean
    ) / (
        std + 1e-7
    )


# ============================================================
# CACHE EMBEDDINGS
# ============================================================

@torch.inference_mode()
def create_embeddings(
    csv_file,
    output_file,
    split_name
):

    # --------------------------------------------------------
    # USE EXISTING CACHE
    # --------------------------------------------------------

    if output_file.exists():

        print(
            f"✓ {split_name} cache already exists:"
            f" {output_file}"
        )

        data = torch.load(
            output_file,
            map_location="cpu",
            weights_only=True,
        )

        return (
            data["embeddings"],
            data["labels"]
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = CREMADAudioDataset(
        csv_file,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=EMBED_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print()
    print("-" * 70)
    print(f"CACHING {split_name}")
    print("-" * 70)

    print(f"Clips: {len(dataset)}")
    print(f"Batches: {len(loader)}")

    all_embeddings = []
    all_labels = []

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):

        audio = batch["audio"].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch["label"]

        audio = normalize_audio(
            audio
        )

        # ----------------------------------------------------
        # WavLM
        # ----------------------------------------------------

        outputs = wavlm(
            input_values=audio
        )

        hidden_states = outputs.last_hidden_state

        # ----------------------------------------------------
        # MEAN POOL
        # ----------------------------------------------------

        embeddings = hidden_states.mean(
            dim=1
        )

        # Move to CPU immediately.
        all_embeddings.append(
            embeddings.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            batch_idx + 1
        ) % 25 == 0 or (
            batch_idx + 1
        ) == len(loader):

            elapsed = (
                time.time()
                - start_time
            )

            processed = (
                batch_idx + 1
            ) * EMBED_BATCH_SIZE

            processed = min(
                processed,
                len(dataset)
            )

            percent = (
                processed
                / len(dataset)
                * 100
            )

            print(
                f"  {processed:5d}/"
                f"{len(dataset)} "
                f"({percent:5.1f}%) "
                f"| {elapsed:.1f}s"
            )

    embeddings = torch.cat(
        all_embeddings,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
        },
        output_file,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"✓ {split_name} cached"
    )

    print(
        f"Shape: {tuple(embeddings.shape)}"
    )

    print(
        f"Time: {elapsed / 60:.2f} minutes"
    )

    print(
        f"Saved: {output_file}"
    )

    return (
        embeddings,
        labels
    )


# ============================================================
# CREATE / LOAD ALL CACHES
# ============================================================

train_embeddings, train_labels = create_embeddings(
    TRAIN_CSV,
    CACHE_DIR / "train.pt",
    "TRAIN"
)

val_embeddings, val_labels = create_embeddings(
    VAL_CSV,
    CACHE_DIR / "validation.pt",
    "VALIDATION"
)

test_embeddings, test_labels = create_embeddings(
    TEST_CSV,
    CACHE_DIR / "test.pt",
    "TEST"
)


# ============================================================
# FREE WAVLM MEMORY
# ============================================================

del wavlm

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print()
print("=" * 70)
print("EMBEDDING CACHE COMPLETE")
print("=" * 70)

print(
    f"Train embeddings: {tuple(train_embeddings.shape)}"
)

print(
    f"Validation embeddings: "
    f"{tuple(val_embeddings.shape)}"
)

print(
    f"Test embeddings: "
    f"{tuple(test_embeddings.shape)}"
)


# ============================================================
# CLASSIFIER
# ============================================================

class EmotionClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                HIDDEN_SIZE,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(
                128,
                NUM_CLASSES
            ),
        )

    def forward(self, x):
        return self.network(x)


model = EmotionClassifier().to(
    DEVICE
)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    train_embeddings,
    train_labels
)

val_dataset = TensorDataset(
    val_embeddings,
    val_labels
)

test_dataset = TensorDataset(
    test_embeddings,
    test_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CLASSIFIER_BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CLASSIFIER_BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CLASSIFIER_BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets
):

    predictions = np.asarray(
        predictions
    )

    targets = np.asarray(
        targets
    )

    accuracy = np.mean(
        predictions == targets
    )

    f1_scores = []

    for class_id in range(
        NUM_CLASSES
    ):

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
            if tp + fp > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0
        )

        f1_scores.append(
            f1
        )

    macro_f1 = np.mean(
        f1_scores
    )

    return accuracy, macro_f1


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    predictions = []
    targets = []

    total_loss = 0.0
    total_items = 0

    for embeddings, labels in loader:

        embeddings = embeddings.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        logits = model(
            embeddings
        )

        loss = criterion(
            logits,
            labels
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        total_items += labels.size(0)

        preds = logits.argmax(
            dim=1
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        targets.extend(
            labels.cpu().numpy()
        )

    accuracy, macro_f1 = calculate_metrics(
        predictions,
        targets
    )

    return (
        total_loss / total_items,
        accuracy,
        macro_f1,
        predictions,
        targets,
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print()
print("=" * 70)
print("TRAINING CLASSIFIER")
print("=" * 70)

best_val_f1 = -1.0
best_state = None
best_epoch = 0

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss = 0.0
    total_items = 0

    train_predictions = []
    train_targets = []

    for embeddings, labels in train_loader:

        embeddings = embeddings.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            embeddings
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        total_items += labels.size(0)

        preds = logits.argmax(
            dim=1
        )

        train_predictions.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        train_targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    train_loss = (
        total_loss
        / total_items
    )

    train_acc, train_f1 = calculate_metrics(
        train_predictions,
        train_targets
    )

    val_loss, val_acc, val_f1, _, _ = evaluate(
        val_loader
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"| Train Loss {train_loss:.4f} "
        f"| Train Acc {train_acc * 100:.2f}% "
        f"| Train F1 {train_f1:.4f} "
        f"| Val Loss {val_loss:.4f} "
        f"| Val Acc {val_acc * 100:.2f}% "
        f"| Val F1 {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch

        best_state = {
            key: value.detach()
            .cpu()
            .clone()
            for key, value in model.state_dict().items()
        }


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_state
)


# ============================================================
# FINAL TEST
# ============================================================

test_loss, test_acc, test_f1, predictions, targets = evaluate(
    test_loader
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

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


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL CREMA-D WAVLM BASELINE")
print("=" * 70)

print(
    f"Best validation epoch : "
    f"{best_epoch}"
)

print(
    f"Best validation F1    : "
    f"{best_val_f1:.4f}"
)

print()

print(
    f"Test Accuracy         : "
    f"{test_acc:.4f} "
    f"({test_acc * 100:.2f}%)"
)

print(
    f"Test Macro F1         : "
    f"{test_f1:.4f}"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION RESULTS")
print("=" * 70)

predictions_np = np.asarray(
    predictions
)

targets_np = np.asarray(
    targets
)

for class_id in range(
    NUM_CLASSES
):

    tp = np.sum(
        (predictions_np == class_id)
        &
        (targets_np == class_id)
    )

    fp = np.sum(
        (predictions_np == class_id)
        &
        (targets_np != class_id)
    )

    fn = np.sum(
        (predictions_np != class_id)
        &
        (targets_np == class_id)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    support = np.sum(
        targets_np == class_id
    )

    print(
        f"{EMOTIONS[class_id]:10s} "
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

print()

print(
    f"{'':12s}" +
    "".join(
        f"{EMOTIONS[i][:9]:>11s}"
        for i in range(NUM_CLASSES)
    )
)

for i in range(NUM_CLASSES):

    row = "".join(
        f"{confusion[i, j]:>11d}"
        for j in range(NUM_CLASSES)
    )

    print(
        f"{EMOTIONS[i][:10]:>10s}  "
        f"{row}"
    )


# ============================================================
# SAVE CLASSIFIER
# ============================================================

classifier_path = (
    Path("models")
    / "cremad_wavlm_frozen_classifier.pt"
)

classifier_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
    },
    classifier_path,
)

print()
print(
    f"Saved classifier: "
    f"{classifier_path}"
)

print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)