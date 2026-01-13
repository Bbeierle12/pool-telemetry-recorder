param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

python -m venv $VenvPath

& "$VenvPath\\Scripts\\python.exe" -m pip install --upgrade pip
$requirements = Join-Path $PSScriptRoot "..\\requirements.txt"
& "$VenvPath\\Scripts\\python.exe" -m pip install -r $requirements

Write-Host "Virtual environment ready at $VenvPath"
