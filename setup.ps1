Write-Host "Creating AudioSpread Virtual Environment..." -ForegroundColor Cyan
python -m venv venv

# Bypass execution policy temporarily for this process to run the activation script
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\venv\Scripts\Activate.ps1

Write-Host "Installing Dependencies..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install librosa numpy pandas mido
pip install fastapi uvicorn websockets onnxruntime

Write-Host ""
Write-Host "AudioSpread Native Environment Setup Complete!" -ForegroundColor Green
Write-Host "To activate the environment in the future, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
