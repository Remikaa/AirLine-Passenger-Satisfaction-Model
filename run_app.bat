@echo off
echo Checking Python installation...
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not found. Please make sure Python is installed and added to PATH
    echo You can download Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python is installed. Installing required packages...
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn PyQt5 joblib

echo.
echo Running the application...
echo (This may take a few minutes on first run while the model trains)
python project.py

if errorlevel 1 (
    echo.
    echo Application encountered an error. Please check the error message above.
    pause
    exit /b 1
)

pause 