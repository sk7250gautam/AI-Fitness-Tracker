"""
AI Fitness Tracker - final exercise tracking module
Last tutorial module:
1. Visualize data to identify patterns
2. Configure Butterworth low-pass filter
3. Apply/tweak low-pass filter
4. Count repetitions
5. Create benchmark dataframe
6. Evaluate results

This module is deliberately self-contained so it can be imported by app.py
without changing the existing model-training pipeline.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prefer the feature-engineered dataset, then fall back to the processed data.
DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "interim" / "03_data_feature_engineered.pkl",
    PROJECT_ROOT / "data" / "interim" / "02_data_outliers_removed.pkl",
    PROJECT_ROOT / "data" / "interim" / "01_data_processed.pkl",
]

OUTPUT_DIR = PROJECT_ROOT / "reports" / "fitness_tracker"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the first available project dataset."""
    for path in DATA_CANDIDATES:
        if path.exists():
            df = pd.read_pickle(path)
            print(f"Using input file: {path.name}")
            print(f"Dataset shape: {df.shape}")
            return df

    raise FileNotFoundError(
        "No project dataset was found in data/interim. "
        "Expected one of: 01_data_processed.pkl, "
        "02_data_outliers_removed.pkl, 03_data_feature_engineered.pkl"
    )


def get_sensor_columns(df):
    """Return available accelerometer and gyroscope columns."""
    acc = [c for c in ("acc_x", "acc_y", "acc_z") if c in df.columns]
    gyr = [c for c in ("gyr_x", "gyr_y", "gyr_z") if c in df.columns]

    if not acc:
        raise ValueError("No accelerometer columns (acc_x/acc_y/acc_z) found.")

    return acc, gyr


def add_sensor_magnitude(df):
    """Create acceleration and gyroscope magnitude signals."""
    result = df.copy()
    acc, gyr = get_sensor_columns(result)

    result["acc_magnitude"] = np.sqrt((result[acc].astype(float) ** 2).sum(axis=1))

    if gyr:
        result["gyr_magnitude"] = np.sqrt((result[gyr].astype(float) ** 2).sum(axis=1))

    return result


def butter_lowpass(cutoff=3.0, fs=30.0, order=4):
    """Configure a Butterworth low-pass filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    if not 0 < normal_cutoff < 1:
        raise ValueError(
            f"cutoff must be between 0 and Nyquist frequency ({nyquist:g} Hz)."
        )

    return butter(order, normal_cutoff, btype="low", analog=False)


def apply_lowpass_filter(signal, cutoff=3.0, fs=30.0, order=4):
    """Apply the configured Butterworth low-pass filter."""
    values = pd.Series(signal, dtype="float64").interpolate().bfill().ffill().to_numpy()

    if len(values) < max(20, order * 3):
        return values

    b, a = butter_lowpass(cutoff=cutoff, fs=fs, order=order)

    try:
        return filtfilt(b, a, values)
    except ValueError:
        return values


def count_repetitions(
    signal,
    fs=30.0,
    min_rep_seconds=0.8,
    prominence=None,
    distance=None,
):
    """
    Count repetitions from a filtered movement signal.

    The default settings are intentionally conservative to avoid counting
    small sensor fluctuations as repetitions.
    """
    values = np.asarray(signal, dtype=float)

    if len(values) == 0:
        return 0, np.array([], dtype=int)

    if prominence is None:
        prominence = max(float(np.std(values)) * 0.35, 1e-8)

    if distance is None:
        distance = max(1, int(fs * min_rep_seconds))

    peaks, _ = find_peaks(
        values,
        distance=distance,
        prominence=prominence,
    )

    return int(len(peaks)), peaks


def create_benchmark_dataframe(df, signal_col="acc_magnitude", fs=30.0):
    """
    Build the benchmark dataframe requested in the tutorial.

    If a label column exists, each exercise/label is evaluated separately.
    The benchmark contains the number of detected repetitions and basic
    signal statistics.
    """
    if signal_col not in df.columns:
        raise ValueError(f"{signal_col!r} is not available.")

    rows = []

    if "label" in df.columns:
        groups = df.groupby("label", dropna=True)
    else:
        groups = [("all", df)]

    for label, group in groups:
        signal = group[signal_col].astype(float).dropna().to_numpy()

        if len(signal) == 0:
            continue

        filtered = apply_lowpass_filter(signal, fs=fs)
        reps, peaks = count_repetitions(filtered, fs=fs)

        rows.append(
            {
                "label": str(label),
                "samples": int(len(signal)),
                "signal_std": float(np.std(signal)),
                "filtered_std": float(np.std(filtered)),
                "repetitions": int(reps),
                "duration_seconds": float(len(signal) / fs),
            }
        )

    return pd.DataFrame(rows)


def visualize_patterns(df, signal_col="acc_magnitude", n_samples=1500):
    """Create the tutorial's pattern-visualization plot."""
    if signal_col not in df.columns:
        raise ValueError(f"{signal_col!r} is not available.")

    values = df[signal_col].astype(float).dropna().to_numpy()
    values = values[:n_samples]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(values)
    ax.set_title("Acceleration magnitude - movement patterns")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Acceleration magnitude")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = OUTPUT_DIR / "01_sensor_patterns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def visualize_filter(df, signal_col="acc_magnitude", fs=30.0, n_samples=1500):
    """Compare raw and filtered movement signals."""
    values = df[signal_col].astype(float).dropna().to_numpy()[:n_samples]
    filtered = apply_lowpass_filter(values, fs=fs)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(values, alpha=0.45, label="Raw")
    ax.plot(filtered, label="Butterworth low-pass")
    ax.set_title("Raw vs filtered movement signal")
    ax.set_xlabel("Sample")
    ax.set_ylabel(signal_col)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = OUTPUT_DIR / "02_lowpass_filter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluate_results(benchmark):
    """Save and print the benchmark evaluation."""
    if benchmark.empty:
        raise ValueError("Benchmark dataframe is empty.")

    output_csv = OUTPUT_DIR / "benchmark_results.csv"
    benchmark.to_csv(output_csv, index=False)

    print("\nBENCHMARK DATAFRAME")
    print(benchmark.to_string(index=False))

    print("\nEVALUATION")
    print(f"Exercise/label groups: {len(benchmark)}")
    print(f"Total detected repetitions: {int(benchmark['repetitions'].sum())}")
    print(f"Results saved to: {output_csv}")

    return benchmark


def run():
    print("=" * 70)
    print("AI FITNESS TRACKER - FINAL TUTORIAL MODULE")
    print("=" * 70)

    df = load_data()
    df = add_sensor_magnitude(df)

    # 1. Visualize data to identify patterns
    pattern_plot = visualize_patterns(df)
    print(f"\nPattern plot saved to: {pattern_plot}")

    # 2-3. Configure and apply Butterworth low-pass filter
    filter_plot = visualize_filter(df)
    print(f"Filter comparison saved to: {filter_plot}")

    # 4-5. Count repetitions and create benchmark dataframe
    benchmark = create_benchmark_dataframe(df)

    # 6. Evaluate results
    evaluate_results(benchmark)

    print("\n" + "=" * 70)
    print("FINAL FITNESS TRACKER MODULE COMPLETE")
    print("=" * 70)

    return benchmark


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run()
