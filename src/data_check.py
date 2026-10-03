from pathlib import Path
import pandas as pd

RAW_DATA = Path("data/raw")

csv_files = list(RAW_DATA.rglob("*.csv"))

print("=" * 50)
print("AI FITNESS TRACKER - DATA CHECK")
print("=" * 50)

print(f"\nTotal CSV files found: {len(csv_files)}")

if not csv_files:
    print("\nNo CSV files found!")
    print("Check data/raw/")
    raise SystemExit

first_file = csv_files[0]

print(f"\nFirst file:")
print(first_file)

try:
    df = pd.read_csv(first_file)

    print("\nDataset loaded successfully!")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

except Exception as error:
    print("\nError while reading file:")
    print(error)