@echo off
REM Evaluation script launcher for Windows
REM
REM This batch file launches the policy evaluation script using Isaac Lab on Windows.
REM
REM Usage:
REM     scripts\play.bat --task Digit-Velocity-Flat-v0 --checkpoint logs\digit_flat\model_15000.pt
REM     scripts\play.bat --task Digit-Velocity-Flat-v0 --checkpoint logs\digit_flat\model_15000.pt --num_envs 16
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

REM Run the evaluation script using Isaac Lab
echo Starting Digit locomotion evaluation...
echo.

call "%ISAACLAB_PATH%\isaaclab.bat" -p "%PROJECT_DIR%\scripts\play.py" %*

if errorlevel 1 (
    echo.
    echo Evaluation failed with error code %errorlevel%
    exit /b %errorlevel%
)

echo.
echo Evaluation completed successfully!
