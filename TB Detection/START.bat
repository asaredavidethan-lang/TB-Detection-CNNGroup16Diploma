@echo off
echo ==========================================
echo   PulmoScan AI - TB Detection System
echo   Confidence Range: 85-97%
echo ==========================================
echo.
cd /d "%~dp0"

:: Delete broken venv if it exists so we start fresh
if exist "venv" (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Could not create virtual environment.
    echo Make sure Python 3.10+ is installed: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [2/3] Installing dependencies (this may take a few minutes)...
venv\Scripts\pip install --upgrade pip
venv\Scripts\pip install fastapi==0.104.1 "uvicorn[standard]==0.24.0" python-multipart==0.0.6
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\pip install pillow opencv-python pydicom numpy pandas jinja2 aiofiles

if errorlevel 1 (
    echo ERROR: Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting server...
echo.
echo ==========================================
echo   Open browser: http://127.0.0.1:8000
echo   Press Ctrl+C to stop
echo ==========================================
echo.
start http://127.0.0.1:8000
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
