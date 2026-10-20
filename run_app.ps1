# Livestock Health AI - PowerShell Launcher

$pythonExe = "$env:LOCALAPPDATA\Python\bin\python.exe"

if (-not (Test-Path $pythonExe)) {
    $found = Get-Command python -ErrorAction SilentlyContinue
    if ($found) {
        $pythonExe = "python"
    } else {
        $foundPy = Get-Command py -ErrorAction SilentlyContinue
        if ($foundPy) {
            $pythonExe = "py"
        } else {
            Write-Error "Could not find a valid Python executable."
            exit 1
        }
    }
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Livestock Health AI - Cattle Disease Classifier" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Using Python: $pythonExe" -ForegroundColor Yellow
Write-Host "Launching Streamlit application..." -ForegroundColor Cyan

& $pythonExe -m streamlit run app.py
