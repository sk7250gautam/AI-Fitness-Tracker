from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    GroupShuffleSplit
)

from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "interim"

REPORT_DIR = PROJECT_ROOT / "reports" / "models"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("AI FITNESS TRACKER")
print("MODEL TRAINING AND EVALUATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

pkl_files = sorted(DATA_DIR.glob("*.pkl"))

if not pkl_files:
    raise FileNotFoundError(
        f"No .pkl files found in:\n{DATA_DIR}"
    )


feature_files = [
    file for file in pkl_files
    if "feature" in file.name.lower()
]

if feature_files:
    INPUT_FILE = feature_files[-1]
else:
    INPUT_FILE = pkl_files[-1]


print("\nUsing input file:")
print(INPUT_FILE.name)


data = pd.read_pickle(INPUT_FILE)

print(f"Dataset shape: {data.shape}")


# ============================================================
# 2. CHECK LABEL
# ============================================================

if "label" not in data.columns:
    raise ValueError(
        "The dataset must contain a 'label' column."
    )


print("\nLabels before cleaning:")
print(data["label"].value_counts(dropna=False))


# Remove rows where label is missing
data = data.dropna(
    subset=["label"]
).copy()


print("\nDataset shape after removing missing labels:")
print(data.shape)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

DROP_COLUMNS = [
    "label",
    "participant",
    "set",
    "category",
    "elapsed",
    "epoch"
]


feature_columns = [
    column
    for column in data.columns
    if column not in DROP_COLUMNS
    and pd.api.types.is_numeric_dtype(
        data[column]
    )
]


if not feature_columns:
    raise ValueError(
        "No numeric feature columns were found."
    )


X = data[feature_columns].copy()

y = data["label"].copy()


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


X = X.fillna(
    X.median(numeric_only=True)
)


# If any complete column is still NaN
X = X.fillna(0)


print(
    f"\nNumber of features: "
    f"{len(feature_columns)}"
)

print(
    f"Number of samples: "
    f"{len(X)}"
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print(
    f"\nTraining samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples: "
    f"{len(X_test)}"
)


# ============================================================
# 6. FEATURE SUBSETS
# ============================================================

acc_features = [
    column
    for column in feature_columns
    if column.startswith("acc_")
]


gyr_features = [
    column
    for column in feature_columns
    if column.startswith("gyr_")
]


print(
    f"Accelerometer features: "
    f"{len(acc_features)}"
)


print(
    f"Gyroscope features: "
    f"{len(gyr_features)}"
)


print(
    f"Total features: "
    f"{len(feature_columns)}"
)


# ============================================================
# 7. FORWARD FEATURE SELECTION
# ============================================================

print("\n" + "=" * 70)
print("FORWARD FEATURE SELECTION")
print("=" * 70)


# SequentialFeatureSelector requires
# selected features < total features.

if len(feature_columns) <= 1:
    raise ValueError(
        "At least 2 numeric features are required "
        "for forward feature selection."
    )


max_features = min(
    20,
    len(feature_columns) - 1
)


print(
    f"\nSelecting {max_features} features..."
)


selector_estimator = DecisionTreeClassifier(
    random_state=42
)


selector = SequentialFeatureSelector(
    selector_estimator,
    n_features_to_select=max_features,
    direction="forward",
    scoring="accuracy",
    cv=3,
    n_jobs=-1
)


selector.fit(
    X_train,
    y_train
)


selected_features = list(
    X_train.columns[
        selector.get_support()
    ]
)


print("\nSelected features:")

for feature in selected_features:
    print(
        f" - {feature}"
    )


X_train_selected = X_train[
    selected_features
]


X_test_selected = X_test[
    selected_features
]


# ============================================================
# 8. GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("GRID SEARCH AND MODEL SELECTION")
print("=" * 70)


models = {

    "Decision Tree": (

        DecisionTreeClassifier(
            random_state=42
        ),

        {
            "max_depth": [
                None,
                5,
                10,
                20
            ],

            "min_samples_split": [
                2,
                5,
                10
            ],

            "criterion": [
                "gini",
                "entropy"
            ]
        }
    ),


    "Random Forest": (

        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),

        {
            "n_estimators": [
                50,
                100
            ],

            "max_depth": [
                None,
                10,
                20
            ],

            "min_samples_split": [
                2,
                5
            ]
        }
    ),


    "Logistic Regression": (

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                LogisticRegression(
                    max_iter=2000
                )
            )
        ]),

        {
            "model__C": [
                0.1,
                1,
                10
            ]
        }
    )
}


results = []

best_models = {}


for model_name, (
    estimator,
    parameter_grid
) in models.items():

    print(
        f"\nTraining {model_name}..."
    )


    grid = GridSearchCV(
        estimator,
        parameter_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )


    grid.fit(
        X_train_selected,
        y_train
    )


    predictions = grid.predict(
        X_test_selected
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    results.append({
        "model": model_name,
        "accuracy": accuracy
    })


    best_models[
        model_name
    ] = grid


    print(
        f"Best CV score: "
        f"{grid.best_score_:.4f}"
    )


    print(
        f"Test accuracy: "
        f"{accuracy:.4f}"
    )


    print(
        f"Best parameters: "
        f"{grid.best_params_}"
    )


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


print(results_df)


plt.figure(
    figsize=(9, 6)
)


plt.bar(
    results_df["model"],
    results_df["accuracy"]
)


plt.ylabel(
    "Accuracy"
)


plt.xlabel(
    "Model"
)


plt.title(
    "Model Accuracy Comparison"
)


plt.ylim(
    0,
    1
)


plt.xticks(
    rotation=20
)


plt.tight_layout()


plot_file = (
    REPORT_DIR
    / "model_comparison.png"
)


plt.savefig(
    plot_file,
    dpi=200
)


plt.close()


print(
    f"\nModel comparison saved to:"
)

print(plot_file)


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["accuracy"].idxmax(),
    "model"
]


best_model = best_models[
    best_model_name
]


best_predictions = best_model.predict(
    X_test_selected
)


best_accuracy = accuracy_score(
    y_test,
    best_predictions
)


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)


print(
    f"Selected model: "
    f"{best_model_name}"
)


print(
    f"Test accuracy: "
    f"{best_accuracy:.4f}"
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:")


print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)


disp.plot()


plt.title(
    f"Confusion Matrix - "
    f"{best_model_name}"
)


plt.tight_layout()


cm_file = (
    REPORT_DIR
    / "confusion_matrix.png"
)


plt.savefig(
    cm_file,
    dpi=200
)


plt.close()


print(
    f"\nConfusion matrix saved to:"
)


print(cm_file)


# ============================================================
# 13. PARTICIPANT-BASED EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("PARTICIPANT-BASED EVALUATION")
print("=" * 70)


if "participant" not in data.columns:

    print(
        "Participant column not available."
    )

else:

    groups = data["participant"]


    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )


    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )


    X_group_train = X.iloc[
        train_idx
    ]


    X_group_test = X.iloc[
        test_idx
    ]


    y_group_train = y.iloc[
        train_idx
    ]


    y_group_test = y.iloc[
        test_idx
    ]


    print(
        f"Participant-train samples: "
        f"{len(X_group_train)}"
    )


    print(
        f"Participant-test samples: "
        f"{len(X_group_test)}"
    )


    print(
        "\nTrain participants:"
    )


    print(
        sorted(
            groups.iloc[
                train_idx
            ].unique()
        )
    )


    print(
        "\nTest participants:"
    )


    print(
        sorted(
            groups.iloc[
                test_idx
            ].unique()
        )
    )


    # --------------------------------------------------------
    # Selected features
    # --------------------------------------------------------

    X_group_train_selected = (
        X_group_train[
            selected_features
        ]
    )


    X_group_test_selected = (
        X_group_test[
            selected_features
        ]
    )


    # --------------------------------------------------------
    # Retrain best model on participant training data
    # --------------------------------------------------------

    participant_model = (
        best_models[
            best_model_name
        ].best_estimator_
    )


    participant_model.fit(
        X_group_train_selected,
        y_group_train
    )


    participant_predictions = (
        participant_model.predict(
            X_group_test_selected
        )
    )


    participant_accuracy = (
        accuracy_score(
            y_group_test,
            participant_predictions
        )
    )


    print(
        f"\nParticipant-based accuracy: "
        f"{participant_accuracy:.4f}"
    )


    print(
        "\nParticipant classification report:"
    )


    print(
        classification_report(
            y_group_test,
            participant_predictions,
            zero_division=0
        )
    )


    # ========================================================
    # 14. SIMPLE MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("SIMPLER MODEL")
    print("=" * 70)


    simple_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )


    simple_model.fit(
        X_group_train_selected,
        y_group_train
    )


    simple_predictions = (
        simple_model.predict(
            X_group_test_selected
        )
    )


    simple_accuracy = (
        accuracy_score(
            y_group_test,
            simple_predictions
        )
    )


    print(
        f"Simple Decision Tree accuracy: "
        f"{simple_accuracy:.4f}"
    )


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results_file = (
    REPORT_DIR
    / "model_results.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


selected_features_file = (
    REPORT_DIR
    / "selected_features.txt"
)


with open(
    selected_features_file,
    "w",
    encoding="utf-8"
) as file:

    for feature in selected_features:

        file.write(
            f"{feature}\n"
        )


print(
    f"\nResults saved to:"
)

print(results_file)


print(
    f"\nSelected features saved to:"
)

print(selected_features_file)


# ============================================================
# 16. DISCUSSION OF RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DISCUSSION OF RESULTS")
print("=" * 70)


print(
    f"""
The model comparison evaluated several
machine-learning algorithms using the
selected feature subset.

Best model:
{best_model_name}

Random train/test accuracy:
{best_accuracy:.4f}

Selected features:
{len(selected_features)}

The participant-based evaluation tests
whether the model generalizes to participants
that were not used for training.

The simpler Decision Tree provides a more
interpretable baseline using the same
selected features.
"""
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)