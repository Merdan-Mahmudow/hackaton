# Deployment script for ML-Web application (PowerShell)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "Starting deployment process..." -ForegroundColor Green
Write-Host "Project root: $ProjectRoot" -ForegroundColor Cyan

Set-Location $ProjectRoot

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python not found" -ForegroundColor Red
    exit 1
}

# Check if uv is available
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Using uv to run deployment script..." -ForegroundColor Cyan
    uv run python scripts/deploy.py
} else {
    Write-Host "Error: uv is required. Please install it first:" -ForegroundColor Red
    Write-Host "  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
    exit 1
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Deployment failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Deployment completed successfully!" -ForegroundColor Green

