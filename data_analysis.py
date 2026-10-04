from pathlib import Path
import itertools

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# AI FITNESS TRACKER
# DATA VISUALIZATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("AI FITNESS TRACKER")
print("DATA VISUALIZATION")
print("=" * 70)


# ============================================================
# 1. FIND PROCESSED DATASET
# ============================================================

possible_files = [
    INTERIM_DIR / "01_data_processed.pkl",
    PROJECT_ROOT / "data" / "processed" / "fitness_dataset.csv",
    PROJECT_ROOT / "data" / "processed" / "01_data_processed.pkl",
]

data_file = None

for file in possible_files:
    if file.exists():
        data_file = file
        break


if data_file is None:
    raise FileNotFoundError(
        "Processed dataset not found.\n"
        "Expected one of:\n"
        "data/interim/01_data_processed.pkl\n"
        "data/processed/fitness_dataset.csv"
    )


print(f"\nLoading dataset:")
print(data_file)


# ============================================================
# 2. LOAD DATA
# ============================================================

if data_file.suffix.lower() == ".pkl":
    data = pd.read_pickle(data_file)

elif data_file.suffix.lower() == ".csv":
    data = pd.read_csv(data_file)

else:
    raise ValueError("Unsupported dataset format.")


print("\nDataset loaded successfully.")
print(f"Shape: {data.shape}")

print("\nColumns:")
for column in data.columns:
    print(f" - {column}")


# ============================================================
# 3. NORMALIZE COLUMN NAMES
# ============================================================

data.columns = [
    str(column).strip().lower().replace(" ", "_")
    for column in data.columns
]


# ------------------------------------------------------------
# Detect accelerometer columns
# ------------------------------------------------------------

ACC_COLUMNS = []

for column in ["acc_x", "acc_y", "acc_z"]:
    if column in data.columns:
        ACC_COLUMNS.append(column)


# ------------------------------------------------------------
# Detect gyroscope columns
# ------------------------------------------------------------

GYR_COLUMNS = []

for column in ["gyr_x", "gyr_y", "gyr_z"]:
    if column in data.columns:
        GYR_COLUMNS.append(column)


# ------------------------------------------------------------
# Support alternative sensor names
# ------------------------------------------------------------

alternative_columns = {
    "x-axis_(g)": "acc_x",
    "y-axis_(g)": "acc_y",
    "z-axis_(g)": "acc_z",

    "x-axis_(deg/s)": "gyr_x",
    "y-axis_(deg/s)": "gyr_y",
    "z-axis_(deg/s)": "gyr_z",

    "x_axis_g": "acc_x",
    "y_axis_g": "acc_y",
    "z_axis_g": "acc_z",

    "x_axis_deg_s": "gyr_x",
    "y_axis_deg_s": "gyr_y",
    "z_axis_deg_s": "gyr_z",
}


for old_name, new_name in alternative_columns.items():

    if old_name in data.columns and new_name not in data.columns:

        data = data.rename(
            columns={old_name: new_name}
        )


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


print("\nAccelerometer columns:")

if ACC_COLUMNS:
    for column in ACC_COLUMNS:
        print(f" - {column}")
else:
    print(" - None detected")


print("\nGyroscope columns:")

if GYR_COLUMNS:
    for column in GYR_COLUMNS:
        print(f" - {column}")
else:
    print(" - None detected")


# ============================================================
# 4. CREATE COMPATIBILITY COLUMNS
# ============================================================
# Sir ke original project me:
#
# participant
# label
# category
# set
#
# Tumhare dataset me agar ye already hain, unko use karenge.
# Agar nahi hain, existing exercise/source information se
# reasonable fallback banayenge.
# ============================================================


# ------------------------------------------------------------
# LABEL
# ------------------------------------------------------------

if "label" not in data.columns:

    if "exercise" in data.columns:

        data["label"] = data["exercise"]

    elif "predicted_exercise" in data.columns:

        data["label"] = data["predicted_exercise"]

    else:

        data["label"] = "unknown"


# ------------------------------------------------------------
# CATEGORY
# ------------------------------------------------------------

if "category" not in data.columns:

    data["category"] = data["label"].astype(str)


# ------------------------------------------------------------
# PARTICIPANT
# ------------------------------------------------------------

if "participant" not in data.columns:

    if "source_file" in data.columns:

        # Try extracting participant-like number
        extracted = (
            data["source_file"]
            .astype(str)
            .str.extract(r"(?:participant|p)[_-]?(\d+)", expand=False)
        )

        data["participant"] = extracted.fillna("unknown")

    else:

        data["participant"] = "unknown"


# ------------------------------------------------------------
# SET
# ------------------------------------------------------------

if "set" not in data.columns:

    if "source_file" in data.columns:

        source = data["source_file"].astype(str).str.lower()

        data["set"] = "unknown"

        data.loc[
            source.str.contains("heavy", na=False),
            "set"
        ] = "heavy"

        data.loc[
            source.str.contains("medium", na=False),
            "set"
        ] = "medium"

        data.loc[
            source.str.contains("light", na=False),
            "set"
        ] = "light"

    else:

        data["set"] = "unknown"


print("\nReference-compatible columns available:")

for column in [
    "participant",
    "label",
    "category",
    "set"
]:
    print(f" - {column}")


# ============================================================
# 5. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATA INFORMATION")
print("=" * 70)

print(f"\nRows       : {len(data):,}")
print(f"Columns    : {len(data.columns)}")
print(f"Memory     : {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


print("\nExercise / label distribution:")

print(
    data["label"]
    .value_counts(dropna=False)
)


print("\nSet distribution:")

print(
    data["set"]
    .value_counts(dropna=False)
)


print("\nParticipant distribution:")

print(
    data["participant"]
    .value_counts(dropna=False)
)


# ============================================================
# 6. REMOVE INVALID SENSOR VALUES FOR VISUALIZATION
# ============================================================

sensor_columns = ACC_COLUMNS + GYR_COLUMNS

if sensor_columns:

    for column in sensor_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data[sensor_columns] = data[sensor_columns].replace(
        [float("inf"), float("-inf")],
        pd.NA
    )


# ============================================================
# 7. PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.grid": True,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_plot(filename):
    """
    Save current matplotlib figure.
    """

    output_path = PLOTS_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved plot: {output_path}")


# ============================================================
# 8. PLOT SINGLE SENSOR COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("SINGLE COLUMN PLOTS")
print("=" * 70)


if sensor_columns:

    sample_size = min(5000, len(data))

    plot_data = data.head(sample_size)

    for column in sensor_columns:

        plt.figure()

        plt.plot(
            plot_data.index,
            plot_data[column],
            linewidth=0.8
        )

        plt.title(
            f"{column} - Sensor Signal"
        )

        plt.xlabel("Sample")
        plt.ylabel(column)

        save_plot(
            f"single_{column}.png"
        )


# ============================================================
# 9. PLOT ALL EXERCISES
# ============================================================

print("\n" + "=" * 70)
print("EXERCISE PLOTS")
print("=" * 70)


if ACC_COLUMNS:

    for exercise in sorted(
        data["label"]
        .dropna()
        .astype(str)
        .unique()
    ):

        exercise_data = data[
            data["label"].astype(str) == exercise
        ].head(5000)

        if exercise_data.empty:
            continue

        plt.figure()

        for column in ACC_COLUMNS:

            plt.plot(
                exercise_data.index,
                exercise_data[column],
                label=column,
                linewidth=0.8
            )

        plt.title(
            f"Accelerometer Signal - {exercise}"
        )

        plt.xlabel("Sample")
        plt.ylabel("Acceleration")

        plt.legend()

        save_plot(
            f"exercise_{exercise}_accelerometer.png"
        )


# ============================================================
# 10. COMPARE MEDIUM VS HEAVY
# ============================================================

print("\n" + "=" * 70)
print("MEDIUM VS HEAVY COMPARISON")
print("=" * 70)


available_sets = set(
    data["set"]
    .dropna()
    .astype(str)
    .str.lower()
)


if "medium" in available_sets and "heavy" in available_sets:

    for column in ACC_COLUMNS:

        medium = data[
            data["set"].astype(str).str.lower() == "medium"
        ][column].dropna().head(5000)

        heavy = data[
            data["set"].astype(str).str.lower() == "heavy"
        ][column].dropna().head(5000)

        plt.figure()

        plt.plot(
            medium.index,
            medium.values,
            label="Medium",
            linewidth=0.8
        )

        plt.plot(
            heavy.index,
            heavy.values,
            label="Heavy",
            linewidth=0.8
        )

        plt.title(
            f"Medium vs Heavy - {column}"
        )

        plt.xlabel("Sample")
        plt.ylabel(column)

        plt.legend()

        save_plot(
            f"medium_vs_heavy_{column}.png"
        )

else:

    print(
        "Medium/heavy labels were not found in the current dataset."
    )


# ============================================================
# 11. COMPARE PARTICIPANTS
# ============================================================

print("\n" + "=" * 70)
print("PARTICIPANT COMPARISON")
print("=" * 70)


participants = (
    data["participant"]
    .dropna()
    .astype(str)
    .unique()
)


if len(participants) > 1 and ACC_COLUMNS:

    selected_participants = participants[:5]

    for column in ACC_COLUMNS:

        plt.figure()

        for participant in selected_participants:

            participant_data = data[
                data["participant"].astype(str)
                == participant
            ][column].dropna().head(3000)

            if participant_data.empty:
                continue

            plt.plot(
                participant_data.index,
                participant_data.values,
                label=f"Participant {participant}",
                linewidth=0.8
            )

        plt.title(
            f"Participant Comparison - {column}"
        )

        plt.xlabel("Sample")
        plt.ylabel(column)

        plt.legend()

        save_plot(
            f"participants_{column}.png"
        )

else:

    print(
        "Multiple participants were not detected."
    )


# ============================================================
# 12. PLOT MULTIPLE ACCELEROMETER AXES
# ============================================================

print("\n" + "=" * 70)
print("MULTI-AXIS ACCELEROMETER PLOTS")
print("=" * 70)


if len(ACC_COLUMNS) >= 2:

    plot_data = data.head(5000)

    plt.figure()

    for column in ACC_COLUMNS:

        plt.plot(
            plot_data.index,
            plot_data[column],
            label=column,
            linewidth=0.8
        )

    plt.title(
        "Accelerometer - All Axes"
    )

    plt.xlabel("Sample")
    plt.ylabel("Acceleration")

    plt.legend()

    save_plot(
        "accelerometer_all_axes.png"
    )


# ============================================================
# 13. PLOT MULTIPLE GYROSCOPE AXES
# ============================================================

print("\n" + "=" * 70)
print("MULTI-AXIS GYROSCOPE PLOTS")
print("=" * 70)


if len(GYR_COLUMNS) >= 2:

    plot_data = data.head(5000)

    plt.figure()

    for column in GYR_COLUMNS:

        plt.plot(
            plot_data.index,
            plot_data[column],
            label=column,
            linewidth=0.8
        )

    plt.title(
        "Gyroscope - All Axes"
    )

    plt.xlabel("Sample")
    plt.ylabel("Angular Velocity")

    plt.legend()

    save_plot(
        "gyroscope_all_axes.png"
    )


# ============================================================
# 14. SENSOR AXIS COMBINATIONS
# ============================================================

print("\n" + "=" * 70)
print("SENSOR AXIS COMBINATIONS")
print("=" * 70)


def create_combinations(columns):

    return list(
        itertools.combinations(columns, 2)
    )


# ------------------------------------------------------------
# Accelerometer combinations
# ------------------------------------------------------------

if len(ACC_COLUMNS) >= 2:

    combinations = create_combinations(
        ACC_COLUMNS
    )

    for combination in combinations:

        plt.figure()

        for column in combination:

            plot_data = data[column].dropna().head(5000)

            plt.plot(
                plot_data.index,
                plot_data.values,
                label=column,
                linewidth=0.8
            )

        plt.title(
            "Accelerometer Comparison: "
            + " vs ".join(combination)
        )

        plt.xlabel("Sample")
        plt.ylabel("Acceleration")

        plt.legend()

        filename = (
            "combination_acc_"
            + "_".join(combination)
            + ".png"
        )

        save_plot(filename)


# ------------------------------------------------------------
# Gyroscope combinations
# ------------------------------------------------------------

if len(GYR_COLUMNS) >= 2:

    combinations = create_combinations(
        GYR_COLUMNS
    )

    for combination in combinations:

        plt.figure()

        for column in combination:

            plot_data = data[column].dropna().head(5000)

            plt.plot(
                plot_data.index,
                plot_data.values,
                label=column,
                linewidth=0.8
            )

        plt.title(
            "Gyroscope Comparison: "
            + " vs ".join(combination)
        )

        plt.xlabel("Sample")
        plt.ylabel("Angular Velocity")

        plt.legend()

        filename = (
            "combination_gyr_"
            + "_".join(combination)
            + ".png"
        )

        save_plot(filename)


# ============================================================
# 15. COMBINE ACCELEROMETER + GYROSCOPE
# ============================================================

print("\n" + "=" * 70)
print("COMBINED SENSOR PLOT")
print("=" * 70)


if ACC_COLUMNS and GYR_COLUMNS:

    plot_data = data.head(5000)

    plt.figure(figsize=(16, 8))

    for column in ACC_COLUMNS:

        plt.plot(
            plot_data.index,
            plot_data[column],
            label=f"ACC - {column}",
            linewidth=0.7
        )

    for column in GYR_COLUMNS:

        plt.plot(
            plot_data.index,
            plot_data[column],
            label=f"GYR - {column}",
            linewidth=0.7
        )

    plt.title(
        "Accelerometer + Gyroscope Signals"
    )

    plt.xlabel("Sample")
    plt.ylabel("Sensor Value")

    plt.legend(
        ncol=2
    )

    save_plot(
        "combined_accelerometer_gyroscope.png"
    )


# ============================================================
# 16. EXPORT SUMMARY
# ============================================================

summary = {
    "rows": len(data),
    "columns": len(data.columns),
    "accelerometer_columns": ", ".join(ACC_COLUMNS),
    "gyroscope_columns": ", ".join(GYR_COLUMNS),
    "number_of_exercises": data["label"].nunique(),
    "number_of_participants": data["participant"].nunique(),
    "number_of_sets": data["set"].nunique(),
}


summary_df = pd.DataFrame(
    [summary]
)

summary_file = REPORTS_DIR / "data_visualization_summary.csv"

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("DATA VISUALIZATION COMPLETE")
print("=" * 70)

print(f"\nDataset:")
print(data_file)

print(f"\nPlots saved to:")
print(PLOTS_DIR)

print(f"\nSummary saved to:")
print(summary_file)

print("\nTotal plots generated:")

plot_count = len(
    list(PLOTS_DIR.glob("*.png"))
)

print(plot_count)

print("\nAll visualization tasks completed successfully.")

print("\n" + "=" * 70)
print("NEXT PART READY")
print("=" * 70)