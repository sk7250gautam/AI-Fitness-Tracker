from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.neighbors import LocalOutlierFactor


# ============================================================
# AI FITNESS TRACKER
# OUTLIER DETECTION & TREATMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "01_data_processed.pkl"

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
REPORT_DIR = PROJECT_ROOT / "reports" / "outliers"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


SENSOR_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AI FITNESS TRACKER")
print("OUTLIER DETECTION & TREATMENT")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_FILE}"
    )

data = pd.read_pickle(INPUT_FILE)

print("\nDataset loaded successfully.")
print(f"Shape: {data.shape}")

print("\nColumns:")
for column in data.columns:
    print(f" - {column}")


# ============================================================
# KEEP ONLY AVAILABLE SENSOR COLUMNS
# ============================================================

sensor_columns = [
    column
    for column in SENSOR_COLUMNS
    if column in data.columns
]

if not sensor_columns:
    raise ValueError(
        "No accelerometer/gyroscope columns found."
    )

print("\nSensor columns used:")

for column in sensor_columns:
    print(f" - {column}")


# Make sure sensor data is numeric
for column in sensor_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================================
# 1. UNDERSTANDING OUTLIERS
# ============================================================

print("\n" + "=" * 70)
print("1. OUTLIER OVERVIEW")
print("=" * 70)

print(
    """
An outlier is an observation that is unusually far away
from the normal pattern of the data.

In fitness sensor data, outliers can occur because of:

- Sensor noise
- Sudden sensor movement
- Incorrect sensor placement
- Communication errors
- Measurement errors
- Extreme but genuine movements

We detect them before modelling so that abnormal sensor
values do not unnecessarily influence the machine-learning
pipeline.
"""
)


# ============================================================
# 2. BOX PLOTS + IQR
# ============================================================

print("\n" + "=" * 70)
print("2. BOXPLOTS AND INTERQUARTILE RANGE")
print("=" * 70)


def calculate_iqr_bounds(series):
    """
    Calculate IQR-based lower and upper limits.
    """

    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return q1, q3, iqr, lower_bound, upper_bound


iqr_summary = []


for column in sensor_columns:

    q1, q3, iqr, lower, upper = calculate_iqr_bounds(
        data[column]
    )

    iqr_summary.append(
        {
            "feature": column,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
        }
    )


iqr_summary_df = pd.DataFrame(iqr_summary)

iqr_summary_df.to_csv(
    REPORT_DIR / "iqr_summary.csv",
    index=False
)


# Boxplot

plt.figure(figsize=(14, 7))

data[sensor_columns].boxplot()

plt.title("Sensor Values - Boxplot")
plt.ylabel("Sensor Value")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "sensor_boxplots.png",
    dpi=150
)

plt.close()

print(
    f"Boxplot saved to: "
    f"{REPORT_DIR / 'sensor_boxplots.png'}"
)


# ============================================================
# 3. MARK OUTLIERS USING IQR
# ============================================================

print("\n" + "=" * 70)
print("3. IQR OUTLIER DETECTION")
print("=" * 70)


def mark_outliers_iqr(df, column):
    """
    Mark observations outside the 1.5 × IQR range.
    """

    _, _, _, lower, upper = calculate_iqr_bounds(
        df[column]
    )

    return (
        (df[column] < lower)
        | (df[column] > upper)
    )


for column in sensor_columns:

    data[f"{column}_outlier_iqr"] = (
        mark_outliers_iqr(data, column)
    )


iqr_outlier_columns = [
    f"{column}_outlier_iqr"]