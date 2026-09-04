import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset import SkinDataset
from models.swin import get_model


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = (
    ROOT_DIR
    / "data"
    / "splits"
    / "PH2"
    / "test.csv"
)

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "optimized"
    / "swin_ph2_optimized.pth"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "results"
    / "ph2_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4

NUM_CLASSES = 3

CLASS_NAMES = [
    "common_nevus",
    "atypical_nevus",
    "melanoma"
]


# ============================================================
# LOAD TEST CSV
# ============================================================

df = pd.read_csv(TEST_CSV)

print("\nTest CSV columns:")
print(df.columns.tolist())

print("\nTest samples:", len(df))


# ============================================================
# DATASET
# ============================================================

dataset = SkinDataset(
    str(TEST_CSV),
    "ph2",
    train=False
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

print("\nUsing device:", DEVICE)

print("Loading optimized Swin:")
print(MODEL_PATH)

model = get_model(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

all_true = []

all_pred = []

all_prob = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_true.extend(
            labels.numpy()
        )

        all_pred.extend(
            predictions.cpu().numpy()
        )

        all_prob.extend(
            probabilities.cpu().numpy()
        )


all_true = np.array(all_true)

all_pred = np.array(all_pred)

all_prob = np.array(all_prob)


# ============================================================
# SAMPLE-LEVEL RESULTS
# ============================================================

results = []

for i in range(len(df)):

    true_label = int(all_true[i])

    predicted_label = int(all_pred[i])

    confidence = float(
        all_prob[i][predicted_label]
    )

    row = df.iloc[i]

    results.append({
        "image": row["image"],
        "image_path": row["image_path"],
        "true_label": true_label,
        "true_class": CLASS_NAMES[true_label],
        "predicted_label": predicted_label,
        "predicted_class": CLASS_NAMES[predicted_label],
        "confidence": confidence,
        "correct": true_label == predicted_label
    })


results_df = pd.DataFrame(results)


# ============================================================
# ERROR ANALYSIS
# ============================================================

errors_df = results_df[
    results_df["correct"] == False
].copy()

errors_df = errors_df.sort_values(
    "confidence",
    ascending=False
)


print("\n" + "=" * 70)
print("PH2 OPTIMIZED SWIN ERROR ANALYSIS")
print("=" * 70)

print(
    "\nTotal test samples:",
    len(results_df)
)

print(
    "Correct predictions:",
    results_df["correct"].sum()
)

print(
    "Incorrect predictions:",
    len(errors_df)
)


# ============================================================
# ERROR TABLE
# ============================================================

print("\nMisclassified samples:")

if len(errors_df) > 0:

    print(
        errors_df[
            [
                "image",
                "true_class",
                "predicted_class",
                "confidence"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print("No errors.")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=list(range(NUM_CLASSES))
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# ERROR TYPES
# ============================================================

if len(errors_df) > 0:

    errors_df["error_type"] = (
        errors_df["true_class"]
        + " -> "
        + errors_df["predicted_class"]
    )

    print("\nError types:")

    print(
        errors_df[
            "error_type"
        ].value_counts().to_string()
    )


# ============================================================
# LOW-CONFIDENCE CORRECT PREDICTIONS
# ============================================================

correct_df = results_df[
    results_df["correct"] == True
].copy()

correct_df = correct_df.sort_values(
    "confidence"
)

print(
    "\nLowest-confidence correct predictions:"
)

print(
    correct_df[
        [
            "image",
            "true_class",
            "confidence"
        ]
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# SAVE ALL PREDICTIONS
# ============================================================

all_predictions_path = (
    OUTPUT_DIR
    / "swin_optimized_all_predictions.csv"
)

results_df.to_csv(
    all_predictions_path,
    index=False
)


# ============================================================
# SAVE ERRORS
# ============================================================

errors_path = (
    OUTPUT_DIR
    / "swin_optimized_errors.csv"
)

errors_df.to_csv(
    errors_path,
    index=False
)


# ============================================================
# SAVE ERROR SUMMARY
# ============================================================

summary = (
    errors_df[
        "error_type"
    ]
    .value_counts()
    .rename_axis("error_type")
    .reset_index(
        name="count"
    )
)

summary_path = (
    OUTPUT_DIR
    / "swin_optimized_error_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)

print("\nFiles saved:")

print(all_predictions_path)

print(errors_path)

print(summary_path)