@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)

if not exist ".env" copy /Y ".env.example" ".env" >nul

echo Starting GeoVision AI at http://127.0.0.1:8000
echo Press Ctrl+C to stop the server.
".venv\Scripts\python.exe" main.py
