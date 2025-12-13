# Azure OpenAI E2E Testing

End-to-end tests for Azure OpenAI endpoints via LiteLLM proxy covering batches, chat completions, messages, and responses APIs.

## 🚀 Setup

### 1. Environment Variables

Create `.env.test` in repository root:

```bash
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-azure-api-key-here
AZURE_API_MODEL="o4-mini" # Ensure model works with Responses API

```

### 2. Start LiteLLM Proxy

```bash
litellm --config azure_testing_config.yaml --port 4000
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install pytest requests python-dotenv
```

## 🏃‍♂️ Running Tests

```bash
# Run all tests (~5-6 minutes)
pytest -v .

# Run specific test files
pytest -v test_e2e_azure_batches.py
pytest -v test_e2e_azure_chat_completions.py
```

## ⏱️ Test Duration

- **Full suite**: ~5-6 minutes  
- **Batch E2E**: ~70-90 seconds (times out at 5 minutes)

## 🔧 Authentication

Tests use `Bearer sk-1234` which matches the `master_key` in `azure_testing_config.yaml`. The proxy forwards requests to Azure using your `AZURE_API_KEY`.

## 🐛 Troubleshooting

**Proxy not running?**
```bash
curl http://localhost:4000/health
```

**Azure auth issues?**
```bash
curl -H "api-key: $AZURE_API_KEY" "$AZURE_API_BASE/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
```
