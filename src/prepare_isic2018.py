import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw/ISIC2018")
OUTPUT_DIR = Path("data/splits/ISIC2018")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = (
    DATA_DIR
    / "ISIC2018_Task3_Training_GroundTruth"
    / "ISIC2018_Task3_Training_GroundTruth.csv"
)

df = pd.read_csv(csv_path)

print("Total images:", len(df))
print("Columns:", df.columns.tolist())

# Convert one-hot labels into one class label
class_columns = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

df["label"] = df[class_columns].idxmax(axis=1)

# Image path
df["image_path"] = df["image"].apply(
    lambda x: str(
        DATA_DIR
        / "ISIC2018_Task3_Training_Input"
        / (x + ".jpg")
    )
)

# Keep image + label + path
df = df[["image", "label", "image_path"]]

# Keep only existing images
df = df[df["image_path"].apply(lambda p: Path(p).exists())]

print("Valid images:", len(df))
print("\nClasses:")
print(df["label"].value_counts())

# 70% train / 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# 15% validation / 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nSplit completed!")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))