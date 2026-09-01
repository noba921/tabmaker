@echo off
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  venv\Scripts\python.exe selftest.py
) else (
  echo [ERROR] venv not found. Run setup.bat first.
  pause
)
