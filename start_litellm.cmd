@echo off
setlocal

cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PATH=%CD%\.venv\Scripts;%PATH%"

if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe
  echo Run this from C:\Users\18747\Desktop\MLops\litellm after the virtual environment is created.
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" ".\start_litellm_detached.py"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo Open: http://localhost:4000/ui/
) else (
  echo LiteLLM failed to start. Check litellm_server.err.log.
)
pause
exit /b %EXIT_CODE%
