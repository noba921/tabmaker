@echo off
cd /d "%~dp0"
title TAB Maker Server

if not exist "venv\Scripts\python.exe" (
  echo [ERROR] venv not found. Run setup.bat first.
  pause
  exit /b 1
)

venv\Scripts\python.exe -c "import flask, numpy, librosa" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Some libraries are missing. Installing...
  venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Install failed. Run setup.bat again, or check.bat to diagnose.
    pause
    exit /b 1
  )
)

echo Starting server... The browser opens automatically when ready.
venv\Scripts\python.exe app.py
echo.
echo Server stopped. If there was an error, see server.log
pause
