🚀 DEEPSEEK-CODER LOCAL - README COMPLETO (QUALQUER PC WINDOWS)
===============================================================================

STATUS ATUAL DO PROJETO (Jan/2026 - Manaus, BR) - SEM QLAUDE.BAT
✅ RTX 5060 8GB + DeepSeek-Coder 6.7B (3.8GB)
✅ 2 Scripts SIMPLES: start-server.bat + deepseek-chat.bat  
✅ VSCode + Continue configurado
✅ Windows Terminal otimizado
✅ 100% offline/local - 40-60 tokens/s JS

📋 PRÉ-REQUISITOS (VERIFICAR)
□ Windows 10/11 atualizado
□ 8GB+ RAM | NVIDIA GPU 6GB+ (RTX 3060+) 
□ 10GB HD livre
□ Internet 1x (só download inicial)

🔧 PASSO A PASSO - INSTALAÇÃO (15 MIN)

1. OLLAMA (3 MIN)
----------------------------------------
Win+R → "https://ollama.com/download"
→ OllamaSetup.exe → ADMIN → Instalar
→ REINICIAR PC OBRIGATÓRIO
----------------------------------------

2. TESTAR OLLAMA (1 MIN)
----------------------------------------
Win+R → "cmd"
ollama --version  (deve mostrar v0.13+)
----------------------------------------

3. BAIXAR MODELO DEEPSEEK (8 MIN - 1X)
----------------------------------------
CMD como ADMIN:
ollama pull deepseek-coder:6.7b  
ollama list  → ✅ 3.8GB carregado
----------------------------------------

🚀 SCRIPTS DO PROJETO (2 ARQUIVOS SIMPLES)

4.1 START-SERVER.BAT (Servidor)
----------------------------------------
@echo off
title 🚀 Ollama Server - DeepSeek Local
echo DEIXE ABERTO! Porta 11434 ativa...
ollama serve
pause
----------------------------------------

4.2 DEEPSEEK-CHAT.BAT (Chat Principal)
----------------------------------------
@echo off
title 🚀 DeepSeek-Coder 6.7B
timeout /t 5 /nobreak >nul
echo 🚀 DeepSeek pronto! Pergunte qualquer coisa:
ollama run deepseek-coder:6.7b
pause
----------------------------------------

🎮 COMO USAR (2 CLIQUES COTIDIANO)
1. start-server.bat → DEIXAR ABERTO (fundo)
2. deepseek-chat.bat → CHAT PRONTO!

⚙️ VSCODE + CONTINUE (BONUS OPCIONAL)

1. Extensions → "Continue" → Instalar
2. Ctrl+, → { } JSON → COLE:
{
  "continue.enableAuthentication": false,
  "continue.telemetryEnabled": false,
  "continue.models": [{
    "title": "DeepSeek Local",
    "provider": "ollama",
    "model": "deepseek-coder:6.7b",
    "apiBase": "http://localhost:11434"
  }]
}

ATALHOS: Ctrl+L (chat) | Ctrl+I (autocomplete JS)

✅ CHECKLIST FINAL (RODAR SEMPRE)
□ ollama list → deepseek-coder:6.7b
□ start-server.bat → "Listening 11434"  
□ deepseek-chat.bat → ">>>" prompt ativo

💾 MIGRAR OUTRO PC (5 MIN)
1. Copiar C:\Users\[USER]\.ollama\models\ → USB
2. Novo PC → Cole MESMA pasta
3. Rodar scripts → ✅ INSTANTÂNEO

🛠️ ERROS COMUNS + SOLUÇÃO
PORTA 11434: start-server.bat (deixar aberto)
CONNECTION REFUSED: Reiniciar PC  
MODELO SUMIU: ollama pull deepseek-coder:6.7b
LENTO: Feche Chrome (5+ abas = 6GB RAM)

📊 CONSUMO RTX 5060
IDLE: 0.5GB VRAM | CHAT: 4.2GB | PIQUE: 6.8GB
JS: 40-60 tokens/s | HTML+CSS: 15-25s

✨ RESULTADO FINAL
💰 GRATIS FOREVER | ⚡ RTX 5060 GPU LOCAL
🎯 JS/NODE/REACT PERFEITO | 🚀 MANAUS OFFLINE
✅ SEM QLAUDE = ZERO ERROS 404

===============================================================================
PROJETO por Gama - Jan/2026 🇧🇷 2 Scripts Simples RTX 5060 DeepSeek-Coder
