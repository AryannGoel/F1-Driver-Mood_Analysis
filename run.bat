@echo off
REM One-command setup + run for Pit Wall (Windows). Double-click, or run in a terminal.
REM Creates an isolated .venv, installs deps (CPU-only torch), starts the app.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment ^(.venv^)...
  python -m venv .venv
)
set "PY=.venv\Scripts\python.exe"

"%PY%" -c "import fastapi, torch, transformers, fastf1, librosa" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies ^(first run - a few hundred MB, a few minutes^)...
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  "%PY%" -m pip install -r requirements.txt
)

where ffmpeg >nul 2>nul || echo note: ffmpeg not found - mp3 usually still decodes; install it if audio fails.
echo.
echo   Pit Wall -^> http://localhost:8000   ^(first analysis downloads ~1 GB of models; Ctrl+C to stop^)
echo.
"%PY%" -m uvicorn app:app --port 8000
