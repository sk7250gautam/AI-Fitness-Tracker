from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np

from flask import Flask, render_template, jsonify, request


# ============================================================
# AI FITNESS TRACKER
# FLASK APPLICATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "models" / "fitness_model.pkl"
DATA_FILE = BASE_DIR / "data" / "processed" / "fitness_dataset.csv"
SESSION_FILE = BASE_DIR / "reports" / "fitness_session.json"
HISTORY_FILE = BASE_DIR / "reports" / "workout_history.json"
REPORT_FILE = BASE_DIR / "reports" / "classification_report.txt"


app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

model_package = None

if MODEL_FILE.exists():
    model_package = joblib.load(MODEL_FILE)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# DASHBOARD DATA
# ============================================================

@app.route("/api/dashboard")
def dashboard():

    result = {
        "model": "Not available",
        "accuracy": 0,
        "exercise": "Unknown",
        "repetitions": 0,
        "total_samples": 0,
        "classes": [],
        "distribution": {}
    }

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    if model_package:

        result["model"] = model_package.get(
            "model_name",
            "Unknown"
        )

        result["accuracy"] = round(
            float(
                model_package.get(
                    "accuracy",
                    0
                )
            ) * 100,
            2
        )

        result["classes"] = model_package.get(
            "classes",
            []
        )


    # --------------------------------------------------------
    # Session information
    # --------------------------------------------------------

    if SESSION_FILE.exists():

        try:

            with open(
                SESSION_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                session = json.load(file)

            result["exercise"] = session.get(
                "detected_exercise",
                "Unknown"
            )

            result["repetitions"] = session.get(
                "estimated_repetitions",
                0
            )

            result["total_samples"] = session.get(
                "total_samples",
                0
            )

            result["distribution"] = session.get(
                "prediction_distribution",
                {}
            )

        except Exception:
            pass


    return jsonify(result)


# ============================================================
# DATASET INFORMATION
# ============================================================

@app.route("/api/dataset")
def dataset():

    if not DATA_FILE.exists():

        return jsonify({
            "error": "Dataset not found"
        }), 404

    try:

        df = pd.read_csv(DATA_FILE)

        exercise_counts = (
            df["exercise"]
            .value_counts()
            .to_dict()
        )

        return jsonify({

            "rows": int(
                len(df)
            ),

            "columns": int(
                len(df.columns)
            ),

            "exercises": {
                str(key): int(value)
                for key, value
                in exercise_counts.items()
            }

        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

@app.route("/api/report")
def report():

    if not REPORT_FILE.exists():

        return jsonify({
            "report": "Classification report not found."
        })

    try:

        report_text = REPORT_FILE.read_text(
            encoding="utf-8"
        )

        return jsonify({
            "report": report_text
        })

    except Exception as error:

        return jsonify({
            "report": str(error)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({
        "status": "running",
        "model_loaded": model_package is not None,
        "python_project": "AI Fitness Tracker"
    })


# ============================================================
# WORKOUT SESSION PERSISTENCE & HISTORY
# ============================================================

@app.route("/api/session/save", methods=["POST"])
def save_session():
    try:
        data = request.get_json() or {}
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)

        session_data = {
            "detected_exercise": data.get("exercise", "Unknown"),
            "estimated_repetitions": data.get("repetitions", 0),
            "total_samples": data.get("total_samples", 0),
            "calories_burned": data.get("calories", 0),
            "form_score": data.get("form_score", 95),
            "timestamp": pd.Timestamp.now().isoformat(),
            "prediction_distribution": data.get("distribution", {})
        }

        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        history = []
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append({
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "exercise": data.get("exercise", "Unknown"),
            "repetitions": data.get("repetitions", 0),
            "calories": data.get("calories", 0),
            "form_score": data.get("form_score", 95),
            "sets": data.get("sets", 1)
        })

        # Retain last 100 historical workouts
        history = history[-100:]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return jsonify({
            "status": "success",
            "message": "Workout session synced and saved successfully!"
        })

    except Exception as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


@app.route("/api/session/history", methods=["GET"])
def get_history():
    if not HISTORY_FILE.exists():
        return jsonify({"history": []})
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        return jsonify({"history": history})
    except Exception as error:
        return jsonify({"history": [], "error": str(error)}), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI FITNESS TRACKER")
    print("=" * 60)

    print("\nStarting Flask server...")
    print("\nOpen in browser:")
    print("http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )