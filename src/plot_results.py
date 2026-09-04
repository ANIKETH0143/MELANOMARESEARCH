import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_FILE = "results/results.csv"
OUTPUT_DIR = Path("results/plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load results
df = pd.read_csv(RESULTS_FILE)

print("\n===== CURRENT RESULTS =====")
print(df.to_string(index=False))


metrics = [
    "Accuracy",
    "Precision",
    "Sensitivity",
    "F1",
    "ROC-AUC"
]


# Create plots
for metric in metrics:

    plt.figure(figsize=(12, 6))

    for dataset in df["Dataset"].unique():

        subset = df[df["Dataset"] == dataset]

        plt.plot(
            subset["Model"],
            subset[metric],
            marker="o",
            label=dataset
        )

    plt.title(f"{metric} Comparison")
    plt.xlabel("Model")
    plt.ylabel(metric)

    plt.ylim(0, 1)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=30)

    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{metric.lower()}_comparison.png"

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved: {output_file}")


print("\n===== PLOT GENERATION COMPLETE =====")