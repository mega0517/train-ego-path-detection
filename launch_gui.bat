@echo off
REM Launch the TEP-Net GUI Application on Windows

REM Get the script directory
set SCRIPT_DIR=%~dp0

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Check and install PyQt5 if needed
python -c "import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo Installing PyQt5...
    python -m pip install PyQt5
)

REM Launch the GUI
cd /d "%SCRIPT_DIR%"
echo Launching TEP-Net GUI Application...
python gui_app.py
pause
