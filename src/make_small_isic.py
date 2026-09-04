import pandas as pd
from pathlib import Path

folder = Path("data/splits/ISIC2018")

for split in ["train", "val", "test"]:

    source = folder / f"{split}.csv"
    output = folder / f"{split}_small.csv"

    df = pd.read_csv(source)

    print(f"\n{split}.csv columns:", df.columns.tolist())

    # Take 50 samples per class
    parts = []

    for label in df["label"].unique():
        class_df = df[df["label"] == label]
        parts.append(
            class_df.sample(
                n=min(50, len(class_df)),
                random_state=42
            )
        )

    small_df = pd.concat(parts, ignore_index=True)

    small_df.to_csv(output, index=False)

    print(f"{split}_small.csv created:", len(small_df))
    print("Columns:", small_df.columns.tolist())