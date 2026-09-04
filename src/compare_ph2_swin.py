from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "results" / "ph2_analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RESULTS
# ============================================================

baseline = {
    "Model": "Swin Baseline",
    "Accuracy": 0.733333,
    "Precision": 0.804029,
    "Sensitivity": 0.694444,
    "F1": 0.719658,
    "ROC-AUC": 0.864198
}


optimized = {
    "Model": "Swin Optimized",
    "Accuracy": 0.700000,
    "Precision": 0.771200,
    "Sensitivity": 0.694444,
    "F1": 0.710000,
    "ROC-AUC": 0.913600
}


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    [baseline, optimized]
)


print("\n" + "=" * 60)
print("PH2 SWIN: BASELINE VS OPTIMIZED")
print("=" * 60)

print(
    df.to_string(
        index=False
    )
)


# ============================================================
# IMPROVEMENT
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Sensitivity",
    "F1",
    "ROC-AUC"
]

comparison = pd.DataFrame({
    "Metric": metrics,
    "Baseline": [
        baseline[m] for m in metrics
    ],
    "Optimized": [
        optimized[m] for m in metrics
    ]
})

comparison["Change"] = (
    comparison["Optimized"]
    - comparison["Baseline"]
)

comparison["Change_Percentage_Points"] = (
    comparison["Change"] * 100
)


print("\nMetric changes:")

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR /
    "swin_baseline_vs_optimized.csv"
)

comparison.to_csv(
    csv_path,
    index=False
)


print(
    "\nSaved:",
    csv_path
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

x = range(len(metrics))

width = 0.35

baseline_values = [
    baseline[m] for m in metrics
]

optimized_values = [
    optimized[m] for m in metrics
]

plt.bar(
    [i - width / 2 for i in x],
    baseline_values,
    width=width,
    label="Baseline"
)

plt.bar(
    [i + width / 2 for i in x],
    optimized_values,
    width=width,
    label="Optimized"
)

plt.xticks(
    list(x),
    metrics
)

plt.ylabel("Score")

plt.xlabel("Metric")

plt.title(
    "PH2 Swin Transformer: Baseline vs Optimized"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()


plot_path = (
    OUTPUT_DIR /
    "swin_baseline_vs_optimized.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    plot_path
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(
    "\nAccuracy change:",
    f"{comparison.loc[0, 'Change_Percentage_Points']:.2f} percentage points"
)

print(
    "F1 change:",
    f"{comparison.loc[3, 'Change_Percentage_Points']:.2f} percentage points"
)

print(
    "ROC-AUC change:",
    f"{comparison.loc[4, 'Change_Percentage_Points']:.2f} percentage points"
)

print(
    "\nThe optimized model substantially improved ROC-AUC, "
    "while accuracy and F1 decreased slightly."
)

print(
    "Therefore, the optimization should be interpreted as "
    "improving ranking/discrimination rather than overall "
    "classification accuracy."
)