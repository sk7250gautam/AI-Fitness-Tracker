from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# AI FITNESS TRACKER
# DATASET CREATION PIPELINE
# ============================================================

RAW_DATA_PATH = Path("data/raw")
INTERIM_DATA_PATH = Path("data/interim")

INTERIM_DATA_PATH.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Extract information from filename
# ------------------------------------------------------------

def extract_filename_features(filename):
    """
    Extract participant, exercise, intensity and set number
    from the MetaMotion filename.
    """

    filename = filename.lower()

    parts = filename.split("_")

    participant = "unknown"
    exercise = "unknown"
    intensity = "unknown"
    set_number = np.nan

    # --------------------------------------------------------
    # Participant
    # --------------------------------------------------------

    if filename.startswith("a-"):
        participant = "A"
    elif filename.startswith("b-"):
        participant = "B"
    elif filename.startswith("c-"):
        participant = "C"
    elif filename.startswith("d-"):
        participant = "D"
    elif filename.startswith("e-"):
        participant = "E"

    # --------------------------------------------------------
    # Exercise
    # --------------------------------------------------------

    if "-bench-" in filename:
        exercise = "bench"

    elif "-squat-" in filename:
        exercise = "squat"

    elif "-ohp-" in filename:
        exercise = "ohp"

    elif "-dead-" in filename:
        exercise = "dead"

    elif "-row-" in filename:
        exercise = "row"

    elif "-rest-" in filename:
        exercise = "rest"

    # --------------------------------------------------------
    # Intensity
    # --------------------------------------------------------

    if "heavy" in filename:
        intensity = "heavy"

    elif "medium" in filename:
        intensity = "medium"

    # --------------------------------------------------------
    # Set number
    # --------------------------------------------------------

    for part in parts:

        if part.isdigit():
            set_number = int(part)
            break

    return participant, exercise, intensity, set_number


# ------------------------------------------------------------
# 2. Read a single CSV file
# ------------------------------------------------------------

def read_single_file(file_path):
    """
    Read one raw MetaMotion CSV file.
    """

    print(f"Reading: {file_path.name}")

    df = pd.read_csv(file_path)

    # Remove completely empty rows
    df = df.dropna(how="all")

    return df


# ------------------------------------------------------------
# 3. Prepare timestamp
# ------------------------------------------------------------

def prepare_timestamp(df):
    """
    Convert epoch milliseconds into a datetime index.
    """

    if "epoch (ms)" in df.columns:

        df["time"] = pd.to_datetime(
            df["epoch (ms)"],
            unit="ms",
            errors="coerce"
        )

    elif "epoch" in df.columns:

        df["time"] = pd.to_datetime(
            df["epoch"],
            unit="ms",
            errors="coerce"
        )

    else:

        raise ValueError(
            "No epoch column found in dataset."
        )

    df = df.dropna(subset=["time"])

    return df


# ------------------------------------------------------------
# 4. Rename sensor columns
# ------------------------------------------------------------

def rename_sensor_columns(df):
    """
    Convert raw sensor column names into consistent names.
    """

    rename_map = {

        "x-axis (g)": "acc_x",
        "y-axis (g)": "acc_y",
        "z-axis (g)": "acc_z",

        "x-axis (deg/s)": "gyr_x",
        "y-axis (deg/s)": "gyr_y",
        "z-axis (deg/s)": "gyr_z",

    }

    df = df.rename(columns=rename_map)

    return df


# ------------------------------------------------------------
# 5. Process one complete file
# ------------------------------------------------------------

def process_file(file_path):

    df = read_single_file(file_path)

    participant, exercise, intensity, set_number = (
        extract_filename_features(file_path.name)
    )

    df = prepare_timestamp(df)

    df = rename_sensor_columns(df)

    # Add metadata
    df["participant"] = participant
    df["label"] = exercise
    df["category"] = intensity
    df["set"] = set_number

    return df


# ------------------------------------------------------------
# 6. Find all CSV files
# ------------------------------------------------------------

def find_csv_files():

    csv_files = list(
        RAW_DATA_PATH.rglob("*.csv")
    )

    print()
    print("=" * 60)
    print("RAW DATA DISCOVERY")
    print("=" * 60)

    print(f"CSV files found: {len(csv_files)}")

    return csv_files


# ------------------------------------------------------------
# 7. Combine all CSV files
# ------------------------------------------------------------

def combine_datasets(csv_files):

    all_data = []

    failed_files = []

    print()
    print("=" * 60)
    print("READING DATASETS")
    print("=" * 60)

    for file_path in csv_files:

        try:

            df = process_file(file_path)

            all_data.append(df)

        except Exception as error:

            failed_files.append(
                (file_path.name, str(error))
            )

            print(
                f"FAILED: {file_path.name}"
            )

            print(
                f"Reason: {error}"
            )

    if not all_data:

        raise RuntimeError(
            "No CSV files could be processed."
        )

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    return combined, failed_files


# ------------------------------------------------------------
# 8. Select useful columns
# ------------------------------------------------------------

def select_columns(df):

    desired_columns = [

        "time",

        "acc_x",
        "acc_y",
        "acc_z",

        "gyr_x",
        "gyr_y",
        "gyr_z",

        "participant",
        "label",
        "category",
        "set",
    ]

    available_columns = [

        column
        for column in desired_columns
        if column in df.columns

    ]

    return df[available_columns]


# ------------------------------------------------------------
# 9. Convert sensor columns to numeric
# ------------------------------------------------------------

def convert_sensor_columns(df):

    sensor_columns = [

        "acc_x",
        "acc_y",
        "acc_z",

        "gyr_x",
        "gyr_y",
        "gyr_z",

    ]

    for column in sensor_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ------------------------------------------------------------
# 10. Remove duplicate timestamps
# ------------------------------------------------------------

def clean_dataset(df):

    df = df.drop_duplicates()

    df = df.sort_values(
        ["participant", "label", "time"]
    )

    return df


# ------------------------------------------------------------
# 11. Resample data to 5 Hz
# ------------------------------------------------------------

def resample_dataset(df):

    print()
    print("=" * 60)
    print("RESAMPLING DATA")
    print("=" * 60)

    numeric_columns = [

        "acc_x",
        "acc_y",
        "acc_z",

        "gyr_x",
        "gyr_y",
        "gyr_z",

    ]

    numeric_columns = [

        column
        for column in numeric_columns
        if column in df.columns

    ]

    results = []

    grouped = df.groupby(
        [
            "participant",
            "label",
            "category",
            "set",
        ],
        dropna=False
    )

    for group_name, group in grouped:

        group = group.copy()

        group = group.set_index("time")

        aggregation = {}

        for column in numeric_columns:

            aggregation[column] = "mean"

        for column in [
            "participant",
            "label",
            "category",
            "set",
        ]:

            if column in group.columns:

                aggregation[column] = "last"

        try:

            resampled = group.resample("200ms").agg(
                aggregation
            )

            results.append(
                resampled.reset_index()
            )

        except Exception as error:

            print(
                f"Resampling failed for {group_name}: {error}"
            )

    if not results:

        raise RuntimeError(
            "Resampling produced no data."
        )

    final_df = pd.concat(
        results,
        ignore_index=True
    )

    return final_df


# ------------------------------------------------------------
# 12. Final cleaning
# ------------------------------------------------------------

def final_cleaning(df):

    numeric_columns = [

        "acc_x",
        "acc_y",
        "acc_z",

        "gyr_x",
        "gyr_y",
        "gyr_z",

    ]

    numeric_columns = [

        column
        for column in numeric_columns
        if column in df.columns

    ]

    # Replace infinite values
    df[numeric_columns] = df[
        numeric_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Interpolate sensor values
    df[numeric_columns] = df[
        numeric_columns
    ].interpolate(
        method="linear",
        limit_direction="both"
    )

    return df


# ------------------------------------------------------------
# 13. Save intermediate dataset
# ------------------------------------------------------------

def save_dataset(df):

    output_file = (
        INTERIM_DATA_PATH
        / "01_data_processed.pkl"
    )

    df.to_pickle(output_file)

    print()
    print("=" * 60)
    print("DATASET SAVED")
    print("=" * 60)

    print(f"Output: {output_file}")
    print(f"Rows  : {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return output_file


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print()
    print("=" * 60)
    print("AI FITNESS TRACKER")
    print("DATASET CREATION PIPELINE")
    print("=" * 60)

    # Find files
    csv_files = find_csv_files()

    if not csv_files:

        raise FileNotFoundError(
            "No CSV files found inside data/raw"
        )

    # Read and combine
    combined, failed_files = combine_datasets(
        csv_files
    )

    print()
    print(
        f"Combined raw shape: {combined.shape}"
    )

    # Select columns
    combined = select_columns(
        combined
    )

    # Numeric conversion
    combined = convert_sensor_columns(
        combined
    )

    # Clean
    combined = clean_dataset(
        combined
    )

    print()
    print(
        f"Cleaned shape: {combined.shape}"
    )

    # Resample
    resampled = resample_dataset(
        combined
    )

    print()
    print(
        f"Resampled shape: {resampled.shape}"
    )

    # Final cleaning
    resampled = final_cleaning(
        resampled
    )

    # Save
    output_file = save_dataset(
        resampled
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print()
    print("Columns:")

    for column in resampled.columns:

        print(f" - {column}")

    if "label" in resampled.columns:

        print()
        print("Exercise distribution:")

        print(
            resampled["label"].value_counts()
        )

    if "participant" in resampled.columns:

        print()
        print("Participant distribution:")

        print(
            resampled["participant"].value_counts()
        )

    print()
    print(
        f"Missing values: "
        f"{resampled.isna().sum().sum()}"
    )

    print()
    print(
        f"Failed files: {len(failed_files)}"
    )

    print()
    print("=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Saved to: {output_file}"
    )


if __name__ == "__main__":

    main()