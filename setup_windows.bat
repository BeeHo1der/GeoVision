@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto :error
)

echo [2/4] Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [3/4] Installing dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [4/4] Preparing .env...
if not exist ".env" copy /Y ".env.example" ".env" >nul

echo.
echo Setup completed.
echo Put best.pth in this folder, then read WINDOWS_MODEL_GUIDE.md.
pause
exit /b 0

:error
echo.
echo Setup failed. Check the error above.
pause
exit /b 1
