import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("data/raw/PH2Dataset")

EXCEL_FILE = BASE_DIR / "PH2_dataset.xlsx"
IMAGE_DIR = BASE_DIR / "PH2 Dataset images"

OUTPUT_DIR = Path("data/splits/PH2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ EXCEL WITHOUT HEADER
# ============================================================

raw = pd.read_excel(
    EXCEL_FILE,
    sheet_name="Folha1",
    header=None
)

print("Excel shape:", raw.shape)


# ============================================================
# PH2 DATA STARTS AT ROW 12
# ============================================================

# Row 11 contains the actual field names:
#
# Image Name
# Histological Diagnosis
# Common Nevus
# Atypical Nevus
# Melanoma
#
# We only need the first 5 columns.

data = raw.iloc[12:].copy()

data = data.iloc[:, :5]

data.columns = [
    "image",
    "histological_diagnosis",
    "common_nevus",
    "atypical_nevus",
    "melanoma"
]

# Remove completely empty rows
data = data.dropna(how="all").reset_index(drop=True)


# ============================================================
# DISPLAY RAW DIAGNOSIS DATA
# ============================================================

print("\nPH2 diagnosis data:")
print(
    data[
        [
            "image",
            "histological_diagnosis",
            "common_nevus",
            "atypical_nevus",
            "melanoma"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# CREATE 3-CLASS LABEL
# ============================================================

def get_label(row):

    if str(row["melanoma"]).strip().upper() == "X":
        return "melanoma"

    if str(row["atypical_nevus"]).strip().upper() == "X":
        return "atypical_nevus"

    if str(row["common_nevus"]).strip().upper() == "X":
        return "common_nevus"

    return None


data["label"] = data.apply(get_label, axis=1)


# Remove rows without a diagnosis
data = data.dropna(subset=["label"]).copy()


# ============================================================
# CREATE IMAGE PATH
# ============================================================

def get_image_path(image_name):

    image_name = str(image_name).strip()

    return (
        IMAGE_DIR
        / image_name
        / f"{image_name}_Dermoscopic_Image"
        / f"{image_name}.bmp"
    )


data["image_path"] = data["image"].apply(get_image_path)


# ============================================================
# CHECK IMAGES
# ============================================================

data["exists"] = data["image_path"].apply(
    lambda x: x.exists()
)

missing = data[~data["exists"]]

if len(missing) > 0:

    print("\nWARNING: Missing images:")
    print(
        missing[
            ["image", "image_path"]
        ].to_string(index=False)
    )

data = data[data["exists"]].copy()

data.drop(columns=["exists"], inplace=True)


# ============================================================
# KEEP REQUIRED COLUMNS
# ============================================================

data = data[
    [
        "image",
        "label",
        "image_path"
    ]
]


# ============================================================
# SHUFFLE
# ============================================================

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# STRATIFIED SPLIT
# ============================================================

# PH2 is small, so we preserve class proportions
# as much as possible.

train_parts = []
val_parts = []
test_parts = []

for label, group in data.groupby("label"):

    group = group.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    n = len(group)

    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    train_parts.append(group.iloc[:train_end])
    val_parts.append(group.iloc[train_end:val_end])
    test_parts.append(group.iloc[val_end:])


train_df = pd.concat(train_parts).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

val_df = pd.concat(val_parts).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

test_df = pd.concat(test_parts).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# SAVE CSV FILES
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "val.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("PH2 DATASET PREPARATION COMPLETE")
print("========================================")

print("\nTotal images:", len(data))

print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nOverall class distribution:")
print(data["label"].value_counts())

print("\nTrain class distribution:")
print(train_df["label"].value_counts())

print("\nValidation class distribution:")
print(val_df["label"].value_counts())

print("\nTest class distribution:")
print(test_df["label"].value_counts())

print("\nFiles created:")

print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "val.csv")
print(OUTPUT_DIR / "test.csv")