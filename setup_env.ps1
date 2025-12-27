$ErrorActionPreference = "Stop"

$venvPath = ".venv"
python -m venv $venvPath

& "$venvPath\\Scripts\\pip" install --upgrade pip
& "$venvPath\\Scripts\\pip" install -r requirements.txt

Write-Host "Environment ready."
Write-Host "Activate with: .\\.venv\\Scripts\\Activate.ps1"
