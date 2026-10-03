@echo off
title AI Fitness Tracker PRO v3.0 Ultra
echo ============================================================
echo Starting AI Fitness Tracker PRO v3.0 Ultra...
echo ============================================================

REM Check if venv exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at venv. Using global Python.
)

REM Run Flask application
python app.py
pause
