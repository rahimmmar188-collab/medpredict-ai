@echo off
TITLE MedPredict AI — Starting Servers...
color 0A

echo.
echo  ==========================================
echo    MedPredict AI — Localhost Launcher
echo  ==========================================
echo.

REM ── Step 1: Start FastAPI Backend on port 8000 ──────────────────────────────
echo  [1/2] Starting FastAPI Backend (port 8000)...
start "MedPredict API Server" cmd /k "cd /d "%~dp0" && uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"

REM ── Give backend time to start ───────────────────────────────────────────────
timeout /t 4 /nobreak > nul

REM ── Step 2: Start Frontend HTTP Server on port 8080 ─────────────────────────
echo  [2/2] Starting Frontend Server (port 8080)...
start "MedPredict Frontend Server" cmd /k "cd /d "%~dp0" && python -m http.server 8080 --directory frontend"

REM ── Give frontend time to start ─────────────────────────────────────────────
timeout /t 2 /nobreak > nul

REM ── Step 3: Open browser ─────────────────────────────────────────────────────
echo.
echo  Opening browser...
start "" "http://localhost:8080"

echo.
echo  ==========================================
echo  MedPredict AI is now running!
echo.
echo  Frontend : http://localhost:8080
echo  API Docs : http://127.0.0.1:8000/docs
echo  Health   : http://127.0.0.1:8000/health
echo  ==========================================
echo.
echo  Close this window to stop all servers, or
echo  close the individual server windows.
echo.
pause
