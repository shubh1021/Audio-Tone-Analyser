from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIG
# ============================================================

METADATA_FILE = Path("data/cremad_metadata.csv")
OUTPUT_DIR = Path("data/processed")

RANDOM_STATE = 42


# ============================================================
# LOAD METADATA
# ============================================================

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_FILE.resolve()}"
    )

df = pd.read_csv(METADATA_FILE)

required_columns = {
    "path",
    "filename",
    "actor_id",
    "sentence",
    "emotion",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================
# SETUP
# ============================================================

X = df["filename"]
y = df["emotion"]
groups = df["actor_id"]


# ============================================================
# FIRST SPLIT
# TEST = ~20%
#
# StratifiedGroupKFold guarantees:
# - actors don't cross the split
# - emotion distribution is considered
# ============================================================

print("=" * 70)
print("CREMA-D SPEAKER-INDEPENDENT SPLIT")
print("=" * 70)

sgkf_test = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

test_indices = None
remaining_indices = None

for fold, (train_idx, test_idx) in enumerate(
    sgkf_test.split(X, y, groups)
):
    # Use the first deterministic fold as test
    if fold == 0:
        test_indices = test_idx
        remaining_indices = train_idx
        break


train_val_df = df.iloc[remaining_indices].copy()
test_df = df.iloc[test_indices].copy()


# ============================================================
# SECOND SPLIT
# VALIDATION = ~16% OF TOTAL
#
# 1/5 of the remaining ~80% = ~16%
# ============================================================

X_train_val = train_val_df["filename"]
y_train_val = train_val_df["emotion"]
groups_train_val = train_val_df["actor_id"]

sgkf_val = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_indices = None
val_indices = None

for fold, (train_idx, val_idx) in enumerate(
    sgkf_val.split(
        X_train_val,
        y_train_val,
        groups_train_val,
    )
):
    if fold == 0:
        train_indices = train_idx
        val_indices = val_idx
        break


train_df = train_val_df.iloc[train_indices].copy()
val_df = train_val_df.iloc[val_indices].copy()


# ============================================================
# RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# VERIFY ACTOR SEPARATION
# ============================================================

train_actors = set(train_df["actor_id"])
val_actors = set(val_df["actor_id"])
test_actors = set(test_df["actor_id"])

train_val_overlap = train_actors & val_actors
train_test_overlap = train_actors & test_actors
val_test_overlap = val_actors & test_actors

if train_val_overlap:
    raise RuntimeError(
        f"Train/Validation actor leakage detected: "
        f"{train_val_overlap}"
    )

if train_test_overlap:
    raise RuntimeError(
        f"Train/Test actor leakage detected: "
        f"{train_test_overlap}"
    )

if val_test_overlap:
    raise RuntimeError(
        f"Validation/Test actor leakage detected: "
        f"{val_test_overlap}"
    )

print()
print("✓ No actor leakage detected.")


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_file = OUTPUT_DIR / "train.csv"
val_file = OUTPUT_DIR / "validation.csv"
test_file = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)


# ============================================================
# REPORT
# ============================================================

def print_split_info(name, data):
    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(f"Clips  : {len(data)}")
    print(f"Actors : {data['actor_id'].nunique()}")

    print()
    print("Emotion distribution:")

    distribution = (
        data["emotion"]
        .value_counts()
        .sort_index()
    )

    for emotion, count in distribution.items():
        percentage = count / len(data) * 100

        print(
            f"{emotion:10s}: "
            f"{count:4d} "
            f"({percentage:5.1f}%)"
        )


print_split_info("TRAIN", train_df)
print_split_info("VALIDATION", val_df)
print_split_info("TEST", test_df)


# ============================================================
# ACTOR LISTS
# ============================================================

print()
print("=" * 70)
print("ACTOR SPLIT")
print("=" * 70)

print()
print(
    f"TRAIN ACTORS ({len(train_actors)}):"
)
print(sorted(train_actors))

print()
print(
    f"VALIDATION ACTORS ({len(val_actors)}):"
)
print(sorted(val_actors))

print()
print(
    f"TEST ACTORS ({len(test_actors)}):"
)
print(sorted(test_actors))


# ============================================================
# FINAL TOTAL
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Total clips      : {len(df)}")
print(f"Training clips   : {len(train_df)}")
print(f"Validation clips : {len(val_df)}")
print(f"Test clips       : {len(test_df)}")

print()
print(f"Train ratio      : {len(train_df) / len(df):.2%}")
print(f"Validation ratio : {len(val_df) / len(df):.2%}")
print(f"Test ratio       : {len(test_df) / len(df):.2%}")

print()
print("Saved files:")
print(f"  {train_file}")
print(f"  {val_file}")
print(f"  {test_file}")

print()
print("=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)