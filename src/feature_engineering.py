from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# AI FITNESS TRACKER
# FEATURE ENGINEERING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "01_data_processed.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "features"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("AI FITNESS TRACKER")
print("FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

data = pd.read_pickle(INPUT_FILE)

print("\nDataset loaded successfully.")
print(f"Shape: {data.shape}")


# ============================================================
# SENSOR COLUMNS
# ============================================================

ACC_COLUMNS = [
    column
    for column in ["acc_x", "acc_y", "acc_z"]
    if column in data.columns
]

GYR_COLUMNS = [
    column
    for column in ["gyr_x", "gyr_y", "gyr_z"]
    if column in data.columns
]

SENSOR_COLUMNS = ACC_COLUMNS + GYR_COLUMNS

if not SENSOR_COLUMNS:
    raise ValueError(
        "No sensor columns were found."
    )

print("\nSensor columns:")

for column in SENSOR_COLUMNS:
    print(f" - {column}")


# ============================================================
# 2. DEALING WITH MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("2. MISSING VALUE IMPUTATION")
print("=" * 70)

for column in SENSOR_COLUMNS:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

    missing_before = int(
        data[column].isna().sum()
    )

    if missing_before > 0:

        data[column] = (
            data[column]
            .interpolate(
                method="linear",
                limit_direction="both"
            )
            .fillna(
                data[column].median()
            )
        )

    missing_after = int(
        data[column].isna().sum()
    )

    print(
        f"{column}: "
        f"{missing_before} -> {missing_after} missing"
    )


# ============================================================
# 3. CALCULATE SET DURATION
# ============================================================

print("\n" + "=" * 70)
print("3. SET DURATION")
print("=" * 70)

if "label" in data.columns:

    group_columns = ["label"]

    if "participant" in data.columns:
        group_columns.append("participant")

    if "set" in data.columns:
        group_columns.append("set")

    if "elapsed" in data.columns:

        data["elapsed"] = pd.to_numeric(
            data["elapsed"],
            errors="coerce"
        )

        data["set_duration"] = (
            data.groupby(group_columns)["elapsed"]
            .transform(
                lambda x: x.max() - x.min()
            )
        )

        print(
            "Set duration calculated from elapsed time."
        )

    elif "epoch" in data.columns:

        data["epoch"] = pd.to_numeric(
            data["epoch"],
            errors="coerce"
        )

        data["set_duration"] = (
            data.groupby(group_columns)["epoch"]
            .transform(
                lambda x: (
                    x.max() - x.min()
                ) / 1000.0
            )
        )

        print(
            "Set duration calculated from epoch."
        )

    else:

        data["set_duration"] = np.nan

        print(
            "No time column available."
        )

else:

    data["set_duration"] = np.nan

    print(
        "Label column not available."
    )


# ============================================================
# 4. BUTTERWORTH LOW-PASS FILTER
# ============================================================

print("\n" + "=" * 70)
print("4. BUTTERWORTH LOW-PASS FILTER")
print("=" * 70)


def butterworth_lowpass(
    series,
    cutoff=0.05,
    order=4
):
    """
    Apply Butterworth low-pass filter.

    Removes high-frequency noise while preserving
    slower movement patterns.
    """

    values = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .interpolate(
            limit_direction="both"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    if len(values) < 20:
        return values

    try:

        b, a = butter(
            order,
            cutoff,
            btype="low"
        )

        filtered = filtfilt(
            b,
            a,
            values
        )

        return filtered

    except Exception:
        return values


for column in SENSOR_COLUMNS:

    filtered_column = (
        f"{column}_filtered"
    )

    data[filtered_column] = (
        butterworth_lowpass(
            data[column]
        )
    )

    print(
        f"Created: {filtered_column}"
    )


# ============================================================
# 5. PRINCIPAL COMPONENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 70)

if len(SENSOR_COLUMNS) >= 2:

    pca_data = data[
        SENSOR_COLUMNS
    ].copy()

    pca_data = pca_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    pca_data = pca_data.fillna(
        pca_data.median()
    )

    scaler = StandardScaler()

    scaled_sensor_data = (
        scaler.fit_transform(
            pca_data
        )
    )

    n_components = min(
        3,
        len(SENSOR_COLUMNS)
    )

    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    principal_components = (
        pca.fit_transform(
            scaled_sensor_data
        )
    )

    for index in range(
        n_components
    ):

        data[
            f"pca_{index + 1}"
        ] = principal_components[:, index]

    explained_variance = (
        pca.explained_variance_ratio_
    )

    print(
        "Explained variance:"
    )

    for index, value in enumerate(
        explained_variance,
        start=1
    ):

        print(
            f"PC{index}: "
            f"{value * 100:.2f}%"
        )

    pd.DataFrame({
        "component": [
            f"PC{i + 1}"
            for i in range(n_components)
        ],
        "explained_variance": (
            explained_variance
        )
    }).to_csv(
        REPORT_DIR
        / "pca_variance.csv",
        index=False
    )

else:

    print(
        "Not enough sensor columns for PCA."
    )


# ============================================================
# 6. SUM OF SQUARES
# ============================================================

print("\n" + "=" * 70)
print("6. SUM OF SQUARES FEATURES")
print("=" * 70)


def calculate_sum_of_squares(
    dataframe,
    columns
):

    if not columns:
        return np.zeros(
            len(dataframe)
        )

    values = (
        dataframe[columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
        .to_numpy()
    )

    return np.sum(
        values ** 2,
        axis=1
    )


if ACC_COLUMNS:

    data["acc_sum_of_squares"] = (
        calculate_sum_of_squares(
            data,
            ACC_COLUMNS
        )
    )

    print(
        "Created acc_sum_of_squares"
    )


if GYR_COLUMNS:

    data["gyr_sum_of_squares"] = (
        calculate_sum_of_squares(
            data,
            GYR_COLUMNS
        )
    )

    print(
        "Created gyr_sum_of_squares"
    )


# ============================================================
# 7. TEMPORAL ABSTRACTION
# ============================================================

print("\n" + "=" * 70)
print("7. TEMPORAL ABSTRACTION")
print("=" * 70)


def add_temporal_features(
    dataframe,
    columns,
    window=20
):

    for column in columns:

        numeric = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )

        dataframe[
            f"{column}_rolling_mean"
        ] = (
            numeric
            .rolling(
                window=window,
                min_periods=1
            )
            .mean()
        )

        dataframe[
            f"{column}_rolling_std"
        ] = (
            numeric
            .rolling(
                window=window,
                min_periods=1
            )
            .std()
            .fillna(0)
        )

    return dataframe


data = add_temporal_features(
    data,
    SENSOR_COLUMNS
)

print(
    "Rolling mean and rolling standard "
    "deviation features created."
)


# ============================================================
# 8. FREQUENCY ABSTRACTION
# ============================================================

print("\n" + "=" * 70)
print("8. FREQUENCY ABSTRACTION")
print("=" * 70)


def dominant_frequency(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 4:
        return 0.0

    values = (
        values
        - np.mean(values)
    )

    spectrum = np.abs(
        np.fft.rfft(values)
    )

    if len(spectrum) <= 1:
        return 0.0

    peak_index = (
        np.argmax(
            spectrum[1:]
        ) + 1
    )

    return float(
        peak_index / len(values)
    )


# Calculate a frequency feature
# using a rolling window.

frequency_window = 50

for column in SENSOR_COLUMNS:

    numeric = pd.to_numeric(
        data[column],
        errors="coerce"
    ).fillna(0)

    data[
        f"{column}_dominant_frequency"
    ] = (
        numeric
        .rolling(
            frequency_window,
            min_periods=10
        )
        .apply(
            dominant_frequency,
            raw=True
        )
        .fillna(0)
    )

print(
    "Dominant frequency features created."
)


# ============================================================
# 9. DEALING WITH DATA OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("9. DATA OVERLAP")
print("=" * 70)


# Sort data when time information exists.

if "epoch" in data.columns:

    data = data.sort_values(
        "epoch"
    ).reset_index(
        drop=True
    )

elif "elapsed" in data.columns:

    data = data.sort_values(
        "elapsed"
    ).reset_index(
        drop=True
    )


# Remove exact duplicate rows.

rows_before = len(data)

data = data.drop_duplicates(
    keep="first"
).reset_index(
    drop=True
)

rows_after = len(data)

print(
    f"Duplicate rows removed: "
    f"{rows_before - rows_after}"
)


# ============================================================
# 10. CLUSTER FEATURES
# ============================================================

print("\n" + "=" * 70)
print("10. K-MEANS CLUSTER FEATURES")
print("=" * 70)


cluster_columns = []

for column in [
    "acc_sum_of_squares",
    "gyr_sum_of_squares",
    "set_duration"
]:

    if column in data.columns:
        cluster_columns.append(
            column
        )


if len(cluster_columns) >= 2:

    cluster_data = (
        data[cluster_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    scaler = StandardScaler()

    scaled_cluster_data = (
        scaler.fit_transform(
            cluster_data
        )
    )

    n_clusters = min(
        4,
        max(2, len(data) // 100)
    )

    n_clusters = min(
        n_clusters,
        len(data)
    )

    if n_clusters >= 2:

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        data["cluster"] = (
            kmeans.fit_predict(
                scaled_cluster_data
            )
        )

        print(
            f"K-Means clustering completed "
            f"with {n_clusters} clusters."
        )

        pd.DataFrame(
            kmeans.cluster_centers_,
            columns=cluster_columns
        ).to_csv(
            REPORT_DIR
            / "cluster_centers.csv",
            index=False
        )

    else:

        data["cluster"] = 0

else:

    data["cluster"] = 0

    print(
        "Not enough features for clustering."
    )


# ============================================================
# FINAL CLEANUP
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANUP")
print("=" * 70)


# Replace infinite values.

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill numerical NaN values.

numeric_columns = (
    data.select_dtypes(
        include=[np.number]
    ).columns
)

for column in numeric_columns:

    if data[column].isna().any():

        median_value = (
            data[column].median()
        )

        if pd.isna(median_value):
            median_value = 0

        data[column] = (
            data[column]
            .fillna(median_value)
        )


# ============================================================
# SAVE FEATURE-ENGINEERED DATA
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR
    / "03_data_feature_engineered.pkl"
)

data.to_pickle(
    OUTPUT_FILE
)


# Also save CSV for inspection.

CSV_FILE = (
    OUTPUT_DIR
    / "03_data_feature_engineered.csv"
)

data.to_csv(
    CSV_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(
    f"Final dataset shape: {data.shape}"
)

print(
    f"Total features: {len(data.columns)}"
)

print(
    f"Saved pickle:\n{OUTPUT_FILE}"
)

print(
    f"Saved CSV:\n{CSV_FILE}"
)

print(
    f"Reports saved in:\n{REPORT_DIR}"
)

print("\nCreated feature categories:")

print("1. Missing value imputation")
print("2. Set duration")
print("3. Butterworth low-pass filtering")
print("4. PCA")
print("5. Sum of squares")
print("6. Temporal abstraction")
print("7. Frequency abstraction")
print("8. Duplicate/overlap handling")
print("9. K-Means clustering")

print("\n" + "=" * 70)
print("NEXT PART READY")
print("=" * 70)