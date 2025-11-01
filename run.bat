@echo off
setlocal

REM Change to the directory of this script
cd /d "%~dp0"

REM Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)

REM Activate the virtual environment
call ".venv\Scripts\activate.bat"

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Allow optional port override via first argument
if not "%1"=="" set PORT=%1
if "%PORT%"=="" set PORT=5000

echo Starting Flask app on port %PORT%...
python app.py

endlocal