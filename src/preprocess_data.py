from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# AI FITNESS TRACKER
# DATA PREPROCESSING
# --------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("AI FITNESS TRACKER - DATA PREPROCESSING")
print("=" * 60)

# Find all CSV files
files = list(RAW_DIR.rglob("*.csv"))

print(f"\nCSV files found: {len(files)}")

if not files:
    raise FileNotFoundError(
        "No CSV files found inside data/raw"
    )

processed_count = 0
failed_count = 0

all_data = []

for file in files:

    try:
        print(f"\nProcessing: {file.name}")

        # Read CSV
        df = pd.read_csv(file)

        # Remove completely empty rows
        df = df.dropna(how="all")

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Convert only truly numeric columns
        for column in df.columns:

            original_non_null = df[column].notna().sum()

            if original_non_null == 0:
                continue

            converted = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            # Convert only if all existing values are numeric
            if converted.notna().sum() == original_non_null:
                df[column] = converted

        # Add source file information
        df["source_file"] = file.name

        # --------------------------------------------------
        # Identify exercise from filename
        # --------------------------------------------------

        filename = file.stem.lower()

        if "bench" in filename:
            exercise = "bench"

        elif "squat" in filename:
            exercise = "squat"

        elif "dead" in filename:
            exercise = "deadlift"

        elif "ohp" in filename:
            exercise = "overhead_press"

        elif "row" in filename:
            exercise = "row"

        elif "rest" in filename:
            exercise = "rest"

        else:
            exercise = "unknown"

        df["exercise"] = exercise

        # Add dataset
        all_data.append(df)

        processed_count += 1

    except Exception as error:

        failed_count += 1

        print(f"\nSkipped: {file.name}")
        print(f"Reason: {error}")


# --------------------------------------------------
# Combine all datasets
# --------------------------------------------------

if not all_data:
    raise RuntimeError(
        "No CSV files could be processed."
    )

combined = pd.concat(
    all_data,
    ignore_index=True
)


# --------------------------------------------------
# Replace infinite values
# --------------------------------------------------

combined = combined.replace(
    [np.inf, -np.inf],
    np.nan
)


# --------------------------------------------------
# Save combined dataset
# --------------------------------------------------

output_file = (
    PROCESSED_DIR /
    "fitness_dataset.csv"
)

combined.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(
    f"\nSuccessfully processed: "
    f"{processed_count}"
)

print(
    f"Failed files: "
    f"{failed_count}"
)

print(
    f"\nFinal dataset shape: "
    f"{combined.shape}"
)

print("\nExercise distribution:")
print(
    combined["exercise"].value_counts()
)

print("\nMissing values:")
print(
    combined.isnull().sum()
)

print("\nProcessed dataset saved at:")
print(output_file)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)