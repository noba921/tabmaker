@echo off
cd /d "%~dp0"
echo ==================================================
echo  TAB Maker - SETUP
echo  This downloads several GB. It may take 10-30 min.
echo ==================================================

rem ---- path length check (PyTorch DLLs fail on long paths) ----
set "HERE=%~dp0"
set "TAIL=%HERE:~80%"
if defined TAIL (
  echo.
  echo [ERROR] This folder path is TOO LONG for the AI libraries:
  echo         %HERE%
  echo.
  echo         Move this whole folder to a short path first, e.g.:
  echo             C:\tabmaker
  echo         then run setup.bat there.
  echo.
  pause
  exit /b 1
)

rem ---- find Python (prefer 3.12 / 3.11 / 3.10) ----
set "PYCMD="
py -3.12 --version >nul 2>&1 && set "PYCMD=py -3.12"
if not defined PYCMD py -3.11 --version >nul 2>&1 && set "PYCMD=py -3.11"
if not defined PYCMD py -3.10 --version >nul 2>&1 && set "PYCMD=py -3.10"
if not defined PYCMD py -3 --version >nul 2>&1 && set "PYCMD=py -3"
if not defined PYCMD python --version >nul 2>&1 && set "PYCMD=python"
if not defined PYCMD (
  echo [ERROR] Python not found.
  echo         Install Python 3.12 from https://www.python.org/downloads/
  echo         IMPORTANT: check "Add python.exe to PATH" during install.
  pause
  exit /b 1
)
echo Using Python: %PYCMD%
%PYCMD% --version

rem ---- create venv ----
%PYCMD% -m venv venv
if not exist "venv\Scripts\python.exe" (
  echo [ERROR] Failed to create venv.
  pause
  exit /b 1
)

rem ---- install core libraries ----
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Library install failed. Run check.bat to diagnose.
  echo         Note: Python 3.13+ may not be supported. Use 3.10-3.12.
  pause
  exit /b 1
)

rem ---- optional engines; each one degrades gracefully if it fails ----
echo.
echo --------------------------------------------------
echo  [1/3] note-detection AI (basic-pitch)
echo --------------------------------------------------
rem basic-pitch は pkg_resources を使う。setuptools 81+ には含まれないので先に固定する
venv\Scripts\python.exe -m pip install "setuptools<81"
venv\Scripts\python.exe -m pip install "basic-pitch[onnx]"
if errorlevel 1 (
  echo [WARN] onnx build failed. Trying the TensorFlow build instead...
  venv\Scripts\python.exe -m pip install "basic-pitch[tf]"
)
rem 依存の解決で setuptools が上げ直されることがあるため、最後にもう一度固定する
venv\Scripts\python.exe -m pip install "setuptools<81"

echo.
echo --------------------------------------------------
echo  [2/3] beat tracker (Beat This! - ISMIR 2024)
echo  Fixes bar-line detection. Strongly recommended.
echo --------------------------------------------------
venv\Scripts\python.exe -m pip install tqdm einops soxr rotary-embedding-torch
venv\Scripts\python.exe -m pip install beat-this

echo.
echo --------------------------------------------------
echo  [3/3] drum transcription AI (ADTOF-pytorch)
echo  Needs Git. Skipped automatically if Git is missing.
echo --------------------------------------------------
git --version >nul 2>&1
if errorlevel 1 (
  echo [SKIP] Git not found - the app will use the built-in drum detector.
  echo        To enable it later: install Git, then run setup.bat again.
) else (
  venv\Scripts\python.exe -m pip install "git+https://github.com/xavriley/ADTOF-pytorch"
)

rem ---- verify the numpy/scipy ABI actually matches (see requirements.txt) ----
echo.
echo --------------------------------------------------
echo  Verifying the installation...
echo --------------------------------------------------
venv\Scripts\python.exe -c "import librosa, numpy, scipy; print('  librosa OK  numpy', numpy.__version__, ' scipy', scipy.__version__)"
if errorlevel 1 (
  echo [FIX] Import failed - repairing numpy/scipy versions...
  venv\Scripts\python.exe -m pip install --force-reinstall "numpy<2" "scipy<1.14"
  venv\Scripts\python.exe -c "import librosa; print('  librosa OK after repair')"
  if errorlevel 1 (
    echo [ERROR] Still broken. Run check.bat and read the [2] section.
    pause
  )
)
venv\Scripts\python.exe -c "import basic_pitch.inference; print('  basic-pitch OK')" 2>nul
if errorlevel 1 (
  echo [WARN] basic-pitch cannot be imported - the app will use the light mode.
  echo        Run check.bat to see the actual error.
)

echo.
echo ==================================================
echo  Installed engines:
venv\Scripts\python.exe -c "import engines; print('   ' + engines.summary_line())" 2>nul
echo.
echo  Setup complete! Double-click run.bat to start.
echo  (selftest.bat checks that everything works.)
echo ==================================================
pause
