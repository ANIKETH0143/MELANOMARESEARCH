import pandas as pd

file = "data/raw/PH2Dataset/PH2_dataset.xlsx"

# Read without assuming a header
raw = pd.read_excel(
    file,
    sheet_name="Folha1",
    header=None
)

print("Excel shape:", raw.shape)

print("\nRows 10-15:")
print(raw.iloc[10:16].to_string(index=False, header=False))

print("\nPossible diagnosis section:")
print(raw.iloc[11:20, 0:6].to_string(index=False, header=False))