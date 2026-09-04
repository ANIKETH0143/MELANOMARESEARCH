import pandas as pd

file = "data/raw/PH2Dataset/PH2_dataset.xlsx"

excel = pd.ExcelFile(file)

print("Sheets:")
print(excel.sheet_names)

for sheet in excel.sheet_names:
    print("\n==============================")
    print("SHEET:", sheet)
    print("==============================")

    df = pd.read_excel(file, sheet_name=sheet)

    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 15 rows:")
    print(df.head(15).to_string(index=False))