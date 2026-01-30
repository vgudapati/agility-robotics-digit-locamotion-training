@echo off
REM Training script launcher for Windows
REM
REM This batch file launches the training script using Isaac Lab on Windows.
REM
REM Usage:
REM     scripts\train.bat --task Digit-Velocity-Flat-v0 --headless
REM     scripts\train.bat --task Digit-Velocity-Rough-v0 --num_envs 4096 --headless
REM
REM Prerequisites:
REM     - Isaac Sim installed via Omniverse Launcher
REM     - Isaac Lab installed and ISAACLAB_PATH environment variable set
REM     - This project installed: pip install -e source\digit_locomotion

setlocal enabledelayedexpansion

REM Check if ISAACLAB_PATH is set
if not defined ISAACLAB_PATH (
    echo ERROR: ISAACLAB_PATH environment variable is not set.
    echo Please set it to your Isaac Lab installation directory.
    echo Example: set ISAACLAB_PATH=C:\Users\YourName\IsaacLab
    exit /b 1
)

REM Get the directory of this script
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

REM Run the training script using Isaac Lab
echo Starting Digit locomotion training...
echo.

call "%ISAACLAB_PATH%\isaaclab.bat" -p "%PROJECT_DIR%\scripts\train.py" %*

if errorlevel 1 (
    echo.
    echo Training failed with error code %errorlevel%
    exit /b %errorlevel%
)

echo.
echo Training completed successfully!
