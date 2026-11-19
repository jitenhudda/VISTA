@echo off
cd /d "%~dp0"
title VISTA Studio v4 - Document Intelligence
 echo ============================================
 echo       VISTA Studio v4 - Document Scanner
 echo ============================================
 echo.
 echo IMPORTANT: Stop the older VISTA app first using Ctrl+C.
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Python 3.11 not found. Install it from python.org.
        pause
        exit /b 1
    )
)
if not exist ".venv\.vista_v4_ready" (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Package installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo installed>".venv\.vista_v4_ready"
)
".venv\Scripts\python.exe" check_setup.py
 echo.
 echo Opening VISTA at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.fileWatcherType none
if errorlevel 1 echo App stopped. Read the message above.
pause
