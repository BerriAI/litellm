@echo off
cd /d "%~dp0"

if exist .env (
  for /f "tokens=*" %%a in (.env) do set %%a
)

.venv\Scripts\litellm.exe --config litellm_config.yaml --port 4000
