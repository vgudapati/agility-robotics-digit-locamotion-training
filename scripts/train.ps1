# Training script launcher for Windows PowerShell
#
# This PowerShell script launches the training script using Isaac Lab on Windows.
#
# Usage:
#     .\scripts\train.ps1 -Task Digit-Velocity-Flat-v0 -Headless
#     .\scripts\train.ps1 -Task Digit-Velocity-Rough-v0 -NumEnvs 4096 -Headless
#     .\scripts\train.ps1 -Task Digit-Velocity-Flat-v0 -MaxIterations 20000 -Headless
#
# Prerequisites:
#     - Isaac Sim installed via Omniverse Launcher
#     - Isaac Lab installed and ISAACLAB_PATH environment variable set
#     - This project installed: pip install -e source\digit_locomotion

param(
    [string]$Task = "Digit-Velocity-Flat-v0",
    [int]$NumEnvs = 0,
    [int]$MaxIterations = 0,
    [int]$Seed = 42,
    [string]$LogDir = "logs",
    [string]$ExperimentName = "",
    [switch]$Headless,
    [switch]$Resume,
    [string]$LoadRun = "",
    [string]$Checkpoint = ""
)

# Check if ISAACLAB_PATH is set
if (-not $env:ISAACLAB_PATH) {
    Write-Error "ISAACLAB_PATH environment variable is not set."
    Write-Host "Please set it to your Isaac Lab installation directory."
    Write-Host "Example: `$env:ISAACLAB_PATH = 'C:\Users\YourName\IsaacLab'"
    exit 1
}

# Get script and project directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# Build command arguments
$args = @(
    "-p", "$ProjectDir\scripts\train.py",
    "--task", $Task,
    "--seed", $Seed,
    "--log_dir", $LogDir
)

if ($NumEnvs -gt 0) {
    $args += @("--num_envs", $NumEnvs)
}

if ($MaxIterations -gt 0) {
    $args += @("--max_iterations", $MaxIterations)
}

if ($ExperimentName) {
    $args += @("--experiment_name", $ExperimentName)
}

if ($Headless) {
    $args += "--headless"
}

if ($Resume) {
    $args += "--resume"
}

if ($LoadRun) {
    $args += @("--load_run", $LoadRun)
}

if ($Checkpoint) {
    $args += @("--checkpoint", $Checkpoint)
}

# Display training info
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DIGIT LOCOMOTION TRAINING (Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Task: $Task"
Write-Host "Headless: $Headless"
Write-Host "Log Directory: $LogDir"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Run training
$isaaclab = Join-Path $env:ISAACLAB_PATH "isaaclab.bat"
& $isaaclab $args

if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Training completed successfully!" -ForegroundColor Green
