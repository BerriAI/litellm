# 🚀 Deploy LiteLLM na Railway (1 Container + SQLite)

## Configuração Super Simples

### 1️⃣ Variáveis de Ambiente na Railway

Adicione estas 3 variáveis no seu serviço Railway:

```bash
LITELLM_MASTER_KEY=sk-1234567890
LITELLM_SALT_KEY=change-this-to-a-random-string-min-32-chars
DATABASE_URL=file:./local.db
```

**Importante:**
- `LITELLM_MASTER_KEY`: Use qualquer chave que comece com `sk-` (será sua senha de API)
- `LITELLM_SALT_KEY`: Use uma string aleatória de pelo menos 32 caracteres
- `DATABASE_URL`: Deixe como `file:./local.db` para usar SQLite

### 2️⃣ Deploy

A Railway vai:
- Detectar o `Dockerfile` automaticamente
- Buildar a imagem
- Expor a porta 4000
- Iniciar o servidor

### 3️⃣ Testando

Após o deploy, pegue a URL do seu app (ex: `https://seu-app.up.railway.app`) e teste:

**Health Check:**
```bash
curl https://seu-app.up.railway.app/health/liveliness
```

**Acessar Dashboard:**
```
https://seu-app.up.railway.app/ui
```

### 4️⃣ Adicionar modelos (Opcional)

Para testar com modelos reais, adicione mais variáveis de ambiente:

```bash
# Para usar OpenAI
OPENAI_API_KEY=sk-sua-chave-openai

# Para usar Anthropic
ANTHROPIC_API_KEY=sk-ant-sua-chave
```

Depois, adicione modelos pelo dashboard UI ou faça uma chamada direto:

```bash
curl https://seu-app.up.railway.app/chat/completions \
  -H "Authorization: Bearer sk-1234567890" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 🎯 Resumo

1. Conecte o repositório na Railway
2. Adicione as 3 variáveis de ambiente
3. Deploy automático
4. Acesse `/ui` para configurar

**Pronto!** 🎉
