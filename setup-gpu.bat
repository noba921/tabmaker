@echo off
cd /d "%~dp0"
echo ==================================================
echo  TAB Maker - GPU SETUP
echo.
echo  Run this ONLY after you have an NVIDIA GPU.
echo  It replaces PyTorch with the CUDA build and adds
echo  the heavy separation models (RoFormer).
echo  Downloads several GB.
echo ==================================================
echo.

if not exist "venv\Scripts\python.exe" (
  echo [ERROR] venv not found. Run setup.bat first.
  pause
  exit /b 1
)

echo Current PyTorch:
venv\Scripts\python.exe -c "import torch;print('  torch',torch.__version__,'CUDA available:',torch.cuda.is_available())"
echo.
echo Press Ctrl+C to cancel, or
pause

rem ---- CUDA build of PyTorch (cu124 works for most recent GPUs) ----
echo.
echo --------------------------------------------------
echo  [1/2] PyTorch with CUDA
echo --------------------------------------------------
venv\Scripts\python.exe -m pip uninstall -y torch torchaudio torchvision
venv\Scripts\python.exe -m pip cache purge
venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
  echo [ERROR] CUDA PyTorch install failed.
  echo         Check the right command for your GPU at https://pytorch.org/get-started/locally/
  echo         Then re-run setup.bat to restore the CPU build if needed.
  pause
  exit /b 1
)

rem ---- high quality separation models ----
echo.
echo --------------------------------------------------
echo  [2/2] RoFormer separation models (audio-separator)
echo --------------------------------------------------
venv\Scripts\python.exe -m pip install "audio-separator[gpu]"
if errorlevel 1 (
  echo [WARN] GPU build failed. Falling back to the CPU build...
  venv\Scripts\python.exe -m pip install "audio-separator[cpu]"
)

echo.
echo ==================================================
venv\Scripts\python.exe -c "import torch;print('torch',torch.__version__,'CUDA:',torch.cuda.is_available())"
venv\Scripts\python.exe -c "import engines; print(engines.summary_line())" 2>nul
echo.
echo  Done. Open the app, expand the engine settings and
echo  switch to the GPU engines you want to use.
echo ==================================================
pause
