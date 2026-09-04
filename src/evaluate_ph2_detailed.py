import os
import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT_DIR / "src"))

from dataset import SkinDataset
from models import get_model


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16
NUM_CLASSES = 3

CLASS_NAMES = [
    "common_nevus",
    "atypical_nevus",
    "melanoma"
]

MODELS = [
    "efficientnetv2",
    "densenet121",
    "resnet101",
    "convnext",
    "swin"
]

TEST_CSV = (
    ROOT_DIR
    / "data"
    / "splits"
    / "PH2"
    / "test.csv"
)

MODEL_DIR = ROOT_DIR / "models"

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
# DEVICE / PATH INFORMATION
# ============================================================

print("Using device:", DEVICE)
print("PH2 test file:", TEST_CSV)
print("Output directory:", OUTPUT_DIR)


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

print("\nTest samples:", len(test_dataset))


# ============================================================
# RESULTS STORAGE
# ============================================================

all_reports = []


# ============================================================
# EVALUATE EACH MODEL
# ============================================================

for model_name in MODELS:

    print("\n" + "=" * 60)
    print("Evaluating:", model_name)
    print("=" * 60)

    # --------------------------------------------------------
    # MODEL PATH
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{model_name}_ph2.pth"
    )

    if not model_path.exists():

        print("WARNING: Model not found:")
        print(model_path)

        continue

    print("Loading:", model_path)

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = get_model(
        model_name,
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        model_path,
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

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

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
                labels.cpu().numpy()
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

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    print("\nConfusion Matrix:")
    print(cm)

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    print("\nClassification Report:")
    print(report_df)

    # ========================================================
    # ROC-AUC
    # ========================================================

    try:

        auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        )

    except ValueError:

        auc = np.nan

    print("\nMacro ROC-AUC:", auc)

    # ========================================================
    # SAVE CLASS-WISE RESULTS
    # ========================================================

    for class_name in CLASS_NAMES:

        all_reports.append({

            "Model": model_name,

            "Class": class_name,

            "Precision":
                report[class_name]["precision"],

            "Recall":
                report[class_name]["recall"],

            "F1":
                report[class_name]["f1-score"],

            "Support":
                report[class_name]["support"],

            "ROC-AUC":
                auc
        })

    # ========================================================
    # CONFUSION MATRIX PLOT
    # ========================================================

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(cm)

    plt.title(
        f"PH2 Confusion Matrix - {model_name}"
    )

    plt.colorbar()

    plt.xticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
        rotation=30,
        ha="right"
    )

    plt.yticks(
        range(NUM_CLASSES),
        CLASS_NAMES
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    # --------------------------------------------------------
    # ADD VALUES TO CELLS
    # --------------------------------------------------------

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_file
    )


# ============================================================
# SAVE CLASS-WISE RESULTS
# ============================================================

if all_reports:

    results_df = pd.DataFrame(
        all_reports
    )

    csv_file = (
        OUTPUT_DIR
        / "ph2_classification_report.csv"
    )

    results_df.to_csv(
        csv_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("PH2 DETAILED ANALYSIS COMPLETE")
    print("=" * 60)

    print("\nSaved:")
    print(csv_file)

    print("\nClass-wise results:")

    print(
        results_df.to_string(
            index=False
        )
    )

else:

    print(
        "\nNo models were successfully evaluated."
    )