@echo off
setlocal

echo ========================================================
echo  Livestock Health AI - Cattle Disease Classifier
echo ========================================================
echo.

:: Detect Python executable
set "PYTHON_CMD="

:: Try local python binary first
if exist "%LOCALAPPDATA%\Python\bin\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Python\bin\python.exe"
    goto :found_python
)

:: Try standard python in PATH
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python -c "import sys; sys.exit(0)" >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_CMD=python"
        goto :found_python
    )
)

:: Try py launcher
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=py"
    goto :found_python
)

echo ERROR: Could not locate a working Python executable.
echo Please ensure Python is installed and added to PATH.
pause
exit /b 1

:found_python
echo Using Python: %PYTHON_CMD%
echo.
echo Launching Streamlit web application...
echo The app will open in your default browser at http://localhost:8501
echo.
"%PYTHON_CMD%" -m streamlit run app.py
pause
