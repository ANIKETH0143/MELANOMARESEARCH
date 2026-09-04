import pandas as pd

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/splits/HAM10000/{split}.csv")

    n = {"train": 300, "val": 75, "test": 75}[split]

    sample = df.groupby("dx", group_keys=False).apply(
        lambda x: x.sample(
            n=max(1, round(n * len(x) / len(df))),
            random_state=42
        )
    )

    sample.to_csv(
        f"data/splits/HAM10000/{split}_small.csv",
        index=False
    )

print("Small sample created!")