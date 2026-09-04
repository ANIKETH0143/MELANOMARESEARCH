# import sys
# import os
# import torch
# import numpy as np
# import pandas as pd

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score
# )

# from dataset import get_loaders

# from models.efficientnetv2 import get_model as efficientnet_model
# from models.densenet121 import get_model as densenet_model
# from models.resnet101 import get_model as resnet_model
# from models.convnext import get_model as convnext_model
# from models.swin import get_model as swin_model


# # -----------------------------
# # Model name
# # -----------------------------
# if len(sys.argv) < 2:
#     raise ValueError(
#         "Please specify model: efficientnetv2, densenet121, "
#         "resnet101, convnext, or swin"
#     )

# MODEL_NAME = sys.argv[1]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("Using device:", device)
# print("Evaluating:", MODEL_NAME)


# # -----------------------------
# # Data
# # -----------------------------
# _, _, test_loader = get_loaders(batch_size=16)


# # -----------------------------
# # Select model
# # -----------------------------
# if MODEL_NAME == "efficientnetv2":
#     model = efficientnet_model(num_classes=7)

# elif MODEL_NAME == "densenet121":
#     model = densenet_model(num_classes=7)

# elif MODEL_NAME == "resnet101":
#     model = resnet_model(num_classes=7)

# elif MODEL_NAME == "convnext":
#     model = convnext_model(num_classes=7)

# elif MODEL_NAME == "swin":
#     model = swin_model(num_classes=7)

# else:
#     raise ValueError("Unknown model name")


# # -----------------------------
# # Load trained weights
# # -----------------------------
# model_path = f"{MODEL_NAME}_ham_small.pth"

# model.load_state_dict(
#     torch.load(model_path, map_location=device)
# )

# model = model.to(device)
# model.eval()


# # -----------------------------
# # Predictions
# # -----------------------------
# y_true = []
# y_pred = []
# y_prob = []

# with torch.no_grad():

#     for images, labels in test_loader:

#         images = images.to(device)

#         outputs = model(images)

#         probabilities = torch.softmax(outputs, dim=1)
#         predictions = outputs.argmax(dim=1)

#         y_true.extend(labels.numpy())
#         y_pred.extend(predictions.cpu().numpy())
#         y_prob.extend(probabilities.cpu().numpy())


# y_prob = np.array(y_prob)


# # -----------------------------
# # Metrics
# # -----------------------------
# accuracy = accuracy_score(y_true, y_pred)

# precision = precision_score(
#     y_true,
#     y_pred,
#     average="weighted",
#     zero_division=0
# )

# sensitivity = recall_score(
#     y_true,
#     y_pred,
#     average="weighted",
#     zero_division=0
# )

# f1 = f1_score(
#     y_true,
#     y_pred,
#     average="weighted",
#     zero_division=0
# )

# roc_auc = roc_auc_score(
#     y_true,
#     y_prob,
#     multi_class="ovr",
#     average="weighted"
# )


# # -----------------------------
# # Display
# # -----------------------------
# print("\n===== RESULTS =====")
# print("Model       :", MODEL_NAME)
# print(f"Accuracy    : {accuracy:.4f}")
# print(f"Precision   : {precision:.4f}")
# print(f"Sensitivity : {sensitivity:.4f}")
# print(f"F1-Score    : {f1:.4f}")
# print(f"ROC-AUC     : {roc_auc:.4f}")


# # -----------------------------
# # Save results
# # -----------------------------
# os.makedirs("results", exist_ok=True)

# result = pd.DataFrame([{
#     "Dataset": "HAM10000",
#     "Model": MODEL_NAME,
#     "Accuracy": accuracy,
#     "Precision": precision,
#     "Sensitivity": sensitivity,
#     "F1": f1,
#     "ROC-AUC": roc_auc
# }])

# result.to_csv(
#     "results/results.csv",
#     mode="a",
#     header=not os.path.exists("results/results.csv"),
#     index=False
# )

# print("\nResults saved to results/results.csv")





# import sys
# import torch
# import pandas as pd
# from torch.utils.data import DataLoader
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score
# )

# from dataset import ISIC2018Dataset
# from models import get_model


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "efficientnetv2"

# print("Using device:", DEVICE)
# print("Evaluating:", MODEL_NAME)


# # Test dataset
# test_dataset = ISIC2018Dataset(
#     "data/splits/ISIC2018/test_small.csv",
#     train=False
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=32,
#     shuffle=False,
#     num_workers=0
# )


# # Create model
# model = get_model(MODEL_NAME, num_classes=7)

# model_path = f"{MODEL_NAME}_isic2018_small.pth"

# model.load_state_dict(
#     torch.load(model_path, map_location=DEVICE)
# )

# model.to(DEVICE)
# model.eval()


# y_true = []
# y_pred = []
# y_prob = []


# with torch.no_grad():

#     for images, labels in test_loader:

#         images = images.to(DEVICE)

#         outputs = model(images)

#         probabilities = torch.softmax(outputs, dim=1)

#         predictions = torch.argmax(probabilities, dim=1)

#         y_true.extend(labels.numpy())
#         y_pred.extend(predictions.cpu().numpy())
#         y_prob.extend(probabilities.cpu().numpy())


# # Metrics
# accuracy = accuracy_score(y_true, y_pred)

# precision = precision_score(
#     y_true,
#     y_pred,
#     average="macro",
#     zero_division=0
# )

# sensitivity = recall_score(
#     y_true,
#     y_pred,
#     average="macro",
#     zero_division=0
# )

# f1 = f1_score(
#     y_true,
#     y_pred,
#     average="macro",
#     zero_division=0
# )

# roc_auc = roc_auc_score(
#     y_true,
#     y_prob,
#     multi_class="ovr",
#     average="macro"
# )


# print("\n===== ISIC2018 RESULTS =====")

# print("Model       :", MODEL_NAME)
# print("Accuracy    :", round(accuracy, 4))
# print("Precision   :", round(precision, 4))
# print("Sensitivity :", round(sensitivity, 4))
# print("F1-Score    :", round(f1, 4))
# print("ROC-AUC     :", round(roc_auc, 4))


# # Save results
# results_file = "results/results.csv"

# result = pd.DataFrame([{
#     "Dataset": "ISIC2018",
#     "Model": MODEL_NAME,
#     "Accuracy": accuracy,
#     "Precision": precision,
#     "Sensitivity": sensitivity,
#     "F1": f1,
#     "ROC-AUC": roc_auc
# }])

# try:
#     old_results = pd.read_csv(results_file)
#     old_results = old_results[
#         ~(
#             (old_results["Dataset"] == "ISIC2018") &
#             (old_results["Model"].str.lower() == MODEL_NAME.lower())
#         )
#     ]
#     result = pd.concat([old_results, result], ignore_index=True)
# except FileNotFoundError:
#     pass

# result.to_csv(results_file, index=False)

# print("\nResults saved to", results_file)







import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from dataset import SkinDataset

from models.efficientnetv2 import get_model as get_efficientnet
from models.densenet121 import get_model as get_densenet
from models.resnet101 import get_model as get_resnet
from models.convnext import get_model as get_convnext
from models.swin import get_model as get_swin


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# ARGUMENTS
# ============================================================

if len(sys.argv) < 3:

    print("\nUsage:")
    print("python src/evaluate.py <dataset> <model>")

    print("\nDatasets:")
    print("  ham10000")
    print("  isic2018")
    print("  ph2")

    print("\nModels:")
    print("  efficientnetv2")
    print("  densenet121")
    print("  resnet101")
    print("  convnext")
    print("  swin")

    sys.exit()


DATASET = sys.argv[1].lower()
MODEL_NAME = sys.argv[2].lower()


# ============================================================
# CLASSES
# ============================================================

if DATASET in ["ham10000", "isic2018"]:

    NUM_CLASSES = 7

elif DATASET == "ph2":

    NUM_CLASSES = 3

else:

    raise ValueError(
        "Unknown dataset. Choose: "
        "ham10000, isic2018, ph2"
    )


print("Evaluating:", DATASET, MODEL_NAME)
print("Number of classes:", NUM_CLASSES)


# ============================================================
# TEST CSV
# ============================================================

if DATASET == "ham10000":

    test_csv = "data/splits/HAM10000/test.csv"

elif DATASET == "isic2018":

    test_csv = "data/splits/ISIC2018/test.csv"

elif DATASET == "ph2":

    test_csv = "data/splits/PH2/test.csv"


# ============================================================
# DATASET
# ============================================================

test_dataset = SkinDataset(
    test_csv,
    dataset_name=DATASET,
    train=False
)


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

if MODEL_NAME == "efficientnetv2":

    model = get_efficientnet(
        num_classes=NUM_CLASSES
    )

elif MODEL_NAME == "densenet121":

    model = get_densenet(
        num_classes=NUM_CLASSES
    )

elif MODEL_NAME == "resnet101":

    model = get_resnet(
        num_classes=NUM_CLASSES
    )

elif MODEL_NAME == "convnext":

    model = get_convnext(
        num_classes=NUM_CLASSES
    )

elif MODEL_NAME == "swin":

    model = get_swin(
        num_classes=NUM_CLASSES
    )

else:

    raise ValueError(
        "Unknown model. Choose: "
        "efficientnetv2, densenet121, resnet101, "
        "convnext, swin"
    )


model = model.to(DEVICE)


# ============================================================
# MODEL PATH
# ============================================================

model_path = Path(
    f"models/{MODEL_NAME}_{DATASET}.pth"
)


if not model_path.exists():

    raise FileNotFoundError(
        f"Model file not found:\n{model_path}"
    )


print("Loading:", model_path)


checkpoint = torch.load(
    model_path,
    map_location=DEVICE
)


model.load_state_dict(checkpoint)

model.eval()


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []


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


        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


y_true = np.array(all_labels)

y_pred = np.array(all_predictions)

y_prob = np.array(all_probabilities)


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


# ============================================================
# ROC-AUC
# ============================================================

try:

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro"
    )

except ValueError:

    roc_auc = float("nan")


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")

print(
    f"{DATASET.upper()} RESULTS"
)

print("========================================")

print(
    f"Model       : {MODEL_NAME}"
)

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
    f"ROC-AUC     : {roc_auc:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_dir = Path("results")

results_dir.mkdir(
    parents=True,
    exist_ok=True
)


results_file = (
    results_dir / "results.csv"
)


new_result = pd.DataFrame([
    {
        "Dataset": DATASET.upper(),
        "Model": MODEL_NAME,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "F1": f1,
        "ROC-AUC": roc_auc
    }
])


# Append to existing results

if results_file.exists():

    old_results = pd.read_csv(
        results_file
    )

    # Remove previous result for
    # the same dataset/model

    old_results = old_results[
        ~(
            (old_results["Dataset"].astype(str).str.lower() == DATASET)
            &
            (old_results["Model"].astype(str).str.lower() == MODEL_NAME)
        )
    ]

    results = pd.concat(
        [old_results, new_result],
        ignore_index=True
    )

else:

    results = new_result


results.to_csv(
    results_file,
    index=False
)


print(
    f"\nResults saved to {results_file}"
)