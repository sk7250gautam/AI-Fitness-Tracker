# ⚡ AI Fitness Tracker PRO v3.0 Ultra
### Computer Vision, Auto-Detect & Biomechanical Form Analyzer

![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?style=for-the-badge&logo=flask)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%203D-00c853?style=for-the-badge&logo=google)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3-38bdf8?style=for-the-badge&logo=tailwindcss)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

An advanced, real-time AI-powered personal fitness coach and biomechanical form evaluator built with **Flask**, **MediaPipe**, **OpenCV**, and a futuristic **Cyberpunk HUD interface**. Features touchless exercise auto-detection, bilateral limb symmetry tracking, a 10-foot Gym Distance HUD mode, interactive voice coaching, multi-exercise workout circuits, and persistent backend database synchronization.

---

## 🌟 Key Features

* **🎥 Dual-Mode Vision Engine:**
  * **Live Webcam Mode:** Direct 60 FPS hardware-accelerated pose tracking via MediaPipe 33 3D skeletal landmarks.
  * **Demo Simulation Mode:** High-fidelity algorithmic simulation for presentations, demonstrations, or low-light testing.
* **🧠 Touchless Auto-Detect Exercise Mode:**
  * Continuously analyzes body geometry and orientation in real-time.
  * Drops into a squat? Switches to **Squats** automatically. Lies down horizontally? Switches to **Push-ups / Bench**. Curls arms? Switches to **Bicep Curls** — no keyboard or mouse touching needed!
* **⚖️ Bilateral Symmetry Tracker (Left vs. Right Limb Balance):**
  * Computes live left-vs-right joint angle symmetry for arms and legs.
  * Alerts on uneven lifts ($>20^\circ$ asymmetry) to prevent muscular imbalances and joint injury (*"Balance your arms, left side lagging!"*).
* **🖥️ Immersive Gym Distance HUD (10-15 Feet Visibility):**
  * Dedicated full-screen high-contrast HUD with giant 150px glowing neon rep counter, massive form status badges, and large telemetry dials readable from across the gym floor.
* **⚡ Multi-Exercise Workout Circuit (Routine Builder):**
  * Integrated **3-Move Full Body Circuit** (Bicep Curls $\rightarrow$ Barbell Squats $\rightarrow$ Push-ups).
  * Automated transition cues and dynamic rest intervals between circuit steps.
* **📐 Biomechanical Form Correction & Cheat Detection:**
  * **Bicep Curls:** Full range-of-motion validation ($>150^\circ \rightarrow <45^\circ$) + elbow sway & momentum cheat detection.
  * **Barbell Squats:** Parallel depth validation ($<95^\circ$) + Knee Valgus (inward knee collapse) detection.
  * **Push-Ups / Bench Press:** Elbow flare protection angle and chest touch tracking.
  * **Deadlift:** Hip hinge angle and spine neutral lockout verification.
* **🎨 Dynamic HUD & Color-Coded Skeletal Feedback:**
  * **Neon Cyan / Emerald:** Optimal biomechanics and safe posture.
  * **Neon Red Alert:** Form deviation / cheat rep detected with instant on-screen HUD banner.
* **🗣️ Voice AI Personal Coach & Web Audio Synth:**
  * Verbal rep announcements and spoken form cues (*"Push your knees out!"*, *"Keep elbows pinned!"*).
  * Web Audio synthesizer providing high-pitch rep completion chimes, rest countdown beeps, and cheat buzzers.
* **💾 Backend Database Sync & History (`app.py`):**
  * Automatically syncs workout sessions, calories burned, form accuracy, and set logs to `/api/session/save`.
  * Persistent storage in `reports/workout_history.json` and `reports/fitness_session.json`.

---

## 🏗️ Project Architecture

```
AI-Fitness-Tracker/
├── app.py                     # Flask application, REST API & Session DB
├── pose_module.py             # OpenCV & MediaPipe PoseDetector engine
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive project documentation
├── templates/
│   └── index.html             # Full-featured Cyberpunk UI, Gym HUD, FSM & Voice Coach
├── static/
│   └── style.css              # Custom styling & animations
├── models/                    # Serialized machine learning models
├── data/                      # Processed datasets
└── reports/
    ├── fitness_session.json   # Live snapshot of active session
    └── workout_history.json   # Persistent lifetime workout logs
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Navigate to Project Directory
```bash
cd D:/AI-Fitness-Tracker
```

### 3. Activate Virtual Environment
```bash
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
python app.py
```

### 6. Open in Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
* Click **`Start Webcam`** on the top navbar to grant camera access and start exercising in real time!
* Click **`Gym HUD`** to enter the immersive 10-foot distance view.
* Toggle **`AUTO: ON`** to let the AI automatically identify your exercise.

---

## 📋 Exercise Biomechanics Specifications

| Exercise | Primary Target Joint | Contraction Threshold | Extension Threshold | Cheat / Form Check |
| :--- | :--- | :--- | :--- | :--- |
| **Bicep Curls** | Elbow Flexion | $< 45^\circ$ | $> 150^\circ$ | Elbow drift / swinging torso / bilateral asymmetry |
| **Squats** | Knee Flexion | $< 95^\circ$ | $> 160^\circ$ | Knee valgus (knees inward) / shallow depth |
| **Push-ups / Bench** | Elbow Joint | $< 85^\circ$ | $> 155^\circ$ | Elbow flare / uneven arm push |
| **Deadlift** | Hip Hinge | $< 100^\circ$ | $> 165^\circ$ | Lower back curvature / incomplete lockout |

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
