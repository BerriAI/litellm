@echo off
title DeepSeek-Coder 6.7B Chat
echo Conectando com servidor Ollama...

REM Testa se servidor está rodando (5s timeout)
timeout /t 5 /nobreak >nul

echo DeepSeek-Coder pronto! Comece a perguntar:
ollama run deepseek-coder:6.7b
pause
