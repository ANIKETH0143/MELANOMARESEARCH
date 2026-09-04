import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
metadata_path = "data/raw/HAM10000/HAM10000_metadata.tab"
image_dir1 = Path("data/raw/HAM10000/images_part_1")
image_dir2 = Path("data/raw/HAM10000/images_part_2")

# Read metadata
df = pd.read_csv(metadata_path, sep="\t")

# Find image path
def find_image(image_id):
    p1 = image_dir1 / f"{image_id}.jpg"
    p2 = image_dir2 / f"{image_id}.jpg"

    if p1.exists():
        return str(p1)
    elif p2.exists():
        return str(p2)
    return None

df["image_path"] = df["image_id"].apply(find_image)

# Check missing images
print("Missing images:", df["image_path"].isna().sum())

# Split
train, temp = train_test_split(
    df, test_size=0.30, stratify=df["dx"], random_state=42
)

val, test = train_test_split(
    temp, test_size=0.50, stratify=temp["dx"], random_state=42
)

# Save
output_dir = Path("data/splits/HAM10000")
output_dir.mkdir(parents=True, exist_ok=True)

train.to_csv(output_dir / "train.csv", index=False)
val.to_csv(output_dir / "val.csv", index=False)
test.to_csv(output_dir / "test.csv", index=False)

print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))
print("Done!")