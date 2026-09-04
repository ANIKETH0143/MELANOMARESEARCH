import sys
from pathlib import Path

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from torch.utils.data import DataLoader

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
# DEVICE
# ============================================================

print("Using device:", DEVICE)

print("Test CSV:", TEST_CSV)

print("Optimized model:", MODEL_PATH)


# ============================================================
# DATASET
# ============================================================

test_dataset = SkinDataset(
    str(TEST_CSV),
    "ph2",
    train=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# ============================================================
# MODEL
# ============================================================

model = get_model(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOAD OPTIMIZED CHECKPOINT
# ============================================================

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

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch")
    )

    print(
        "Best validation F1:",
        checkpoint.get("best_val_f1")
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()


# ============================================================
# PREDICTION
# ============================================================

y_true = []

y_pred = []

y_prob = []


with torch.no_grad():

    for images, labels in test_loader:

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

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )

        y_prob.extend(
            probabilities.cpu().numpy()
        )


y_true = np.array(y_true)

y_pred = np.array(y_pred)

y_prob = np.array(y_prob)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

sensitivity = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


try:

    auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro"
    )

except ValueError:

    auc = float("nan")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 50)
print("OPTIMIZED SWIN — PH2 TEST RESULTS")
print("=" * 50)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Sensitivity : {sensitivity:.4f}"
)

print(
    f"F1-Score    : {f1:.4f}"
)

print(
    f"ROC-AUC     : {auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nClass order:")

print(CLASS_NAMES)