import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

ERROR_CSV = (
    ROOT_DIR
    / "results"
    / "ph2_analysis"
    / "swin_optimized_errors.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "results"
    / "ph2_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD ERROR DATA
# ============================================================

if not ERROR_CSV.exists():

    print("ERROR CSV not found:")
    print(ERROR_CSV)
    sys.exit(1)


errors = pd.read_csv(ERROR_CSV)

print("\nLoaded errors:", len(errors))
print(errors.to_string(index=False))


# ============================================================
# FIND IMAGE PATH
# ============================================================

# The error CSV may contain image_path.
# If not, use the PH2 test CSV to recover it.

TEST_CSV = (
    ROOT_DIR
    / "data"
    / "splits"
    / "PH2"
    / "test.csv"
)

test_data = pd.read_csv(TEST_CSV)

print("\nTest CSV columns:")
print(test_data.columns.tolist())


if "image_path" in errors.columns:

    errors["resolved_path"] = errors["image_path"]

else:

    # Match image ID with the test CSV
    image_column = None

    for column in ["image", "image_id", "id"]:

        if column in test_data.columns:

            image_column = column
            break

    if image_column is None:

        print("ERROR: Could not find image identifier.")
        sys.exit(1)

    path_map = dict(
        zip(
            test_data[image_column].astype(str),
            test_data["image_path"]
        )
    )

    errors["resolved_path"] = (
        errors["image"]
        .astype(str)
        .map(path_map)
    )


# ============================================================
# VERIFY IMAGE PATHS
# ============================================================

valid_errors = []

for _, row in errors.iterrows():

    image_path = Path(str(row["resolved_path"]))

    if image_path.exists():

        valid_errors.append(row)

    else:

        print(
            "\nWARNING: Image not found:",
            image_path
        )


errors = pd.DataFrame(valid_errors)

print(
    "\nImages available for visualization:",
    len(errors)
)


if len(errors) == 0:

    print("No valid images found.")
    sys.exit(1)


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

n = len(errors)

cols = 3
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(15, 5 * rows)
)

# Make axes always iterable
if rows == 1:

    axes = axes.reshape(1, -1)


for idx, (_, row) in enumerate(errors.iterrows()):

    r = idx // cols
    c = idx % cols

    ax = axes[r, c]

    image_path = Path(
        str(row["resolved_path"])
    )

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        ax.imshow(image)

    except Exception as e:

        ax.text(
            0.5,
            0.5,
            f"Unable to load image\n{e}",
            ha="center",
            va="center"
        )

    ax.axis("off")

    image_id = row.get(
        "image",
        image_path.stem
    )

    true_class = row.get(
        "true_class",
        "unknown"
    )

    predicted_class = row.get(
        "predicted_class",
        "unknown"
    )

    confidence = row.get(
        "confidence",
        None
    )

    if pd.notna(confidence):

        title = (
            f"{image_id}\n"
            f"True: {true_class}\n"
            f"Predicted: {predicted_class}\n"
            f"Confidence: {float(confidence):.3f}"
        )

    else:

        title = (
            f"{image_id}\n"
            f"True: {true_class}\n"
            f"Predicted: {predicted_class}"
        )

    ax.set_title(
        title,
        fontsize=11
    )


# Hide unused axes

for idx in range(n, rows * cols):

    r = idx // cols
    c = idx % cols

    axes[r, c].axis("off")


plt.suptitle(
    "PH2 Optimized Swin — Misclassified Samples",
    fontsize=18
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR
    / "swin_optimized_error_contact_sheet.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\n" + "=" * 60)
print("PH2 ERROR VISUALIZATION COMPLETE")
print("=" * 60)

print("\nSaved:")
print(output_file)