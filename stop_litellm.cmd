@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "PID_FILE=%CD%\litellm_server.pid"
set "PORT=4000"

:: 1. Try to stop by PID file
call :StopByPidFile

:: 2. Fallback: kill any process still listening on port 4000
call :StopByPort

:: 3. Verify
netstat -ano | findstr ":%PORT%" | findstr "LISTENING" >nul
if %ERRORLEVEL%==0 (
    echo ERROR: port %PORT% is still in use.
    exit /b 1
) else (
    echo Stopped. Port %PORT% is free.
)

pause
exit /b 0

:StopByPidFile
if not exist "%PID_FILE%" (
    echo No PID file found at %PID_FILE%
    goto :eof
)
set /p PID=<%PID_FILE%
if "%PID%"=="" (
    echo PID file is empty.
    goto :eof
)
echo Stopping process from PID file: %PID%
taskkill /PID %PID% /F /T >nul 2>&1
if %ERRORLEVEL%==0 (
    echo Process %PID% stopped.
    del "%PID_FILE%"
) else (
    echo Failed to stop process %PID% or it was already gone.
)
goto :eof

:StopByPort
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    echo Stopping process on port %PORT%: PID %%a
    taskkill /PID %%a /F /T >nul 2>&1
    if !ERRORLEVEL!==0 (
        echo Process %%a stopped.
    ) else (
        echo Failed to stop process %%a.
    )
)
goto :eof
