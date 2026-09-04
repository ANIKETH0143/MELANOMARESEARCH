import pandas as pd

files = [
    "data/Splits/HAM10000/train.csv",
    "data/Splits/HAM10000/val.csv",
    "data/Splits/HAM10000/test.csv"
]

print("HAM10000 SUBGROUP METADATA AUDIT")

for f in files:
    df = pd.read_csv(f)

    print("\n" + f)
    print("Missing values:")
    print(df[["sex", "age"]].isna().sum().to_string())

    print("\nSex distribution:")
    print(df["sex"].value_counts(dropna=False).to_string())

    print("\nAge summary:")
    print(df["age"].describe().to_string())
