@echo off
REM Launch script for DocSynthesis-V1 Gradio Demo (Windows)
REM IndiaAI IDP Challenge Submission

echo =========================================
echo 🏆 DocSynthesis-V1 Interactive Demo
echo IndiaAI IDP Challenge 2024
echo =========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update requirements
echo 📥 Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements-gradio.txt

REM Check if full requirements are needed
if "%1"=="--full" (
    echo 📥 Installing full system requirements...
    pip install -q -r requirements.txt
)

echo.
echo =========================================
echo 🚀 Launching Gradio Interface...
echo =========================================
echo.
echo 📱 The demo will be available at:
echo    - Local: http://localhost:7860
echo    - Public: (Gradio will generate a shareable link^)
echo.
echo Press Ctrl+C to stop the server
echo.

REM Launch the Gradio app
python gradio_app.py

pause

