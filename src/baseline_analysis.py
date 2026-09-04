# import pandas as pd
# from pathlib import Path

# RESULTS_FILE = Path("results/results.csv")
# OUTPUT_DIR = Path("results/baseline_analysis")

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# df = pd.read_csv(RESULTS_FILE)

# print("\n" + "=" * 70)
# print("CONSOLIDATED BASELINE ANALYSIS")
# print("=" * 70)

# print("\nAll Results:")
# print(df.to_string(index=False))

# # ------------------------------------------------------------
# # BEST MODEL BY DATASET
# # ------------------------------------------------------------

# best_accuracy = df.loc[
#     df.groupby("Dataset")["Accuracy"].idxmax()
# ].copy()

# best_accuracy = best_accuracy[
#     ["Dataset", "Model", "Accuracy", "Precision",
#      "Sensitivity", "F1", "ROC-AUC"]
# ]

# print("\n" + "=" * 70)
# print("BEST MODEL BY ACCURACY")
# print("=" * 70)

# print(best_accuracy.to_string(index=False))

# best_accuracy.to_csv(
#     OUTPUT_DIR / "best_model_by_dataset.csv",
#     index=False
# )

# # ------------------------------------------------------------
# # BEST MODEL BY F1
# # ------------------------------------------------------------

# best_f1 = df.loc[
#     df.groupby("Dataset")["F1"].idxmax()
# ].copy()

# best_f1 = best_f1[
#     ["Dataset", "Model", "Accuracy", "Precision",
#      "Sensitivity", "F1", "ROC-AUC"]
# ]

# print("\n" + "=" * 70)
# print("BEST MODEL BY F1")
# print("=" * 70)

# print(best_f1.to_string(index=False))

# best_f1.to_csv(
#     OUTPUT_DIR / "best_model_by_f1.csv",
#     index=False
# )

# # ------------------------------------------------------------
# # BEST MODEL BY ROC-AUC
# # ------------------------------------------------------------

# best_auc = df.loc[
#     df.groupby("Dataset")["ROC-AUC"].idxmax()
# ].copy()

# best_auc = best_auc[
#     ["Dataset", "Model", "Accuracy", "Precision",
#      "Sensitivity", "F1", "ROC-AUC"]
# ]

# print("\n" + "=" * 70)
# print("BEST MODEL BY ROC-AUC")
# print("=" * 70)

# print(best_auc.to_string(index=False))

# best_auc.to_csv(
#     OUTPUT_DIR / "best_model_by_auc.csv",
#     index=False
# )

# # ------------------------------------------------------------
# # AVERAGE PERFORMANCE ACROSS DATASETS
# # ------------------------------------------------------------

# numeric_cols = [
#     "Accuracy",
#     "Precision",
#     "Sensitivity",
#     "F1",
#     "ROC-AUC"
# ]

# model_average = (
#     df.groupby("Model")[numeric_cols]
#     .mean()
#     .sort_values("F1", ascending=False)
# )

# print("\n" + "=" * 70)
# print("AVERAGE PERFORMANCE ACROSS ALL DATASETS")
# print("=" * 70)

# print(model_average.to_string())

# model_average.to_csv(
#     OUTPUT_DIR / "model_average_performance.csv"
# )

# # ------------------------------------------------------------
# # DATASET AVERAGES
# # ------------------------------------------------------------

# dataset_average = (
#     df.groupby("Dataset")[numeric_cols]
#     .mean()
# )

# print("\n" + "=" * 70)
# print("AVERAGE PERFORMANCE BY DATASET")
# print("=" * 70)

# print(dataset_average.to_string())

# dataset_average.to_csv(
#     OUTPUT_DIR / "dataset_average_performance.csv"
# )

# # ------------------------------------------------------------
# # COMPLETE SUMMARY
# # ------------------------------------------------------------

# summary = {
#     "Best Accuracy": best_accuracy[
#         ["Dataset", "Model", "Accuracy"]
#     ],
#     "Best F1": best_f1[
#         ["Dataset", "Model", "F1"]
#     ],
#     "Best ROC-AUC": best_auc[
#         ["Dataset", "Model", "ROC-AUC"]
#     ]
# }

# print("\n" + "=" * 70)
# print("ANALYSIS COMPLETE")
# print("=" * 70)

# print("\nFiles saved to:")
# print(OUTPUT_DIR)




import pandas as pd
from pathlib import Path

RESULTS_FILE = Path("results/results.csv")
OUTPUT_DIR = Path("results/baseline_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_FILE)

# Normalize model names for analysis only.
MODEL_NAME_MAP = {
    "efficientnetv2": "EfficientNetV2",
    "EfficientNetV2": "EfficientNetV2",
    "densenet121": "DenseNet121",
    "resnet101": "ResNet101",
    "convnext": "ConvNeXt",
    "swin": "Swin",
}

df["Model"] = df["Model"].map(
    lambda x: MODEL_NAME_MAP.get(str(x), str(x))
)

print("=" * 70)
print("CONSOLIDATED BASELINE ANALYSIS")
print("=" * 70)

print("\nAll Results:")
print(df.to_string(index=False))

# Best model by accuracy
best_accuracy = (
    df.loc[df.groupby("Dataset")["Accuracy"].idxmax()]
    .sort_values("Dataset")
)

print("\n" + "=" * 70)
print("BEST MODEL BY ACCURACY")
print("=" * 70)
print(best_accuracy.to_string(index=False))
best_accuracy.to_csv(
    OUTPUT_DIR / "best_model_by_dataset.csv",
    index=False
)

# Best model by F1
best_f1 = (
    df.loc[df.groupby("Dataset")["F1"].idxmax()]
    .sort_values("Dataset")
)

print("\n" + "=" * 70)
print("BEST MODEL BY F1")
print("=" * 70)
print(best_f1.to_string(index=False))
best_f1.to_csv(
    OUTPUT_DIR / "best_model_by_f1.csv",
    index=False
)

# Best model by ROC-AUC
best_auc = (
    df.loc[df.groupby("Dataset")["ROC-AUC"].idxmax()]
    .sort_values("Dataset")
)

print("\n" + "=" * 70)
print("BEST MODEL BY ROC-AUC")
print("=" * 70)
print(best_auc.to_string(index=False))
best_auc.to_csv(
    OUTPUT_DIR / "best_model_by_auc.csv",
    index=False
)

# Average performance by model
model_average = (
    df.groupby("Model")[
        ["Accuracy", "Precision", "Sensitivity", "F1", "ROC-AUC"]
    ]
    .mean()
    .sort_values("F1", ascending=False)
)

print("\n" + "=" * 70)
print("AVERAGE PERFORMANCE ACROSS ALL DATASETS")
print("=" * 70)
print(model_average.to_string())

model_average.to_csv(
    OUTPUT_DIR / "model_average_performance.csv"
)

# Average performance by dataset
dataset_average = (
    df.groupby("Dataset")[
        ["Accuracy", "Precision", "Sensitivity", "F1", "ROC-AUC"]
    ]
    .mean()
)

print("\n" + "=" * 70)
print("AVERAGE PERFORMANCE BY DATASET")
print("=" * 70)
print(dataset_average.to_string())

dataset_average.to_csv(
    OUTPUT_DIR / "dataset_average_performance.csv"
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nFiles saved to:")
print(OUTPUT_DIR)