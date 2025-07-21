# LiteLLM Built-in MCP Quick Setup Guide

## 🎯 Get Started in 3 Minutes

### 1️⃣ Create Configuration File

Create `my_config.yaml` file:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

mcp_builtin_servers:
  calculator:
    transport: "stdio"
    command: "python"
    args: ["examples/sample_calculator_mcp_server.py"]
    enabled: true

general_settings:
  master_key: sk-1234
```

### 2️⃣ Set Environment Variable

```bash
export OPENAI_API_KEY="your-openai-key"
```

### 3️⃣ Start Server

```bash
litellm --config my_config.yaml
```

### 4️⃣ Test

```bash
curl -X POST http://localhost:4000/responses \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "input": [{"type": "message", "role": "user", "content": "What is 10 + 5?"}],
    "tools": [{"type": "mcp", "builtin": "calculator"}]
  }'
```

## 🔧 Adding More Services

### Add Zapier
```yaml
mcp_builtin_servers:
  zapier:
    url: "https://mcp.zapier.com/api/mcp"
    transport: "sse"
    auth_type: "bearer_token"
    env_key: "ZAPIER_TOKEN"
    enabled: true
```

```bash
export ZAPIER_TOKEN="your-zapier-token"
```

### Add GitHub
```yaml
mcp_builtin_servers:
  github:
    url: "https://mcp.github.com/mcp"
    transport: "http"
    auth_type: "bearer_token"
    env_key: "GITHUB_TOKEN"
    enabled: true
```

```bash
export GITHUB_TOKEN="your-github-token"
```

## 💡 Usage Patterns

### Server Token Usage
```json
{"tools": [{"type": "mcp", "builtin": "zapier"}]}
```

### Client Token Usage
```json
{"tools": [{"type": "mcp", "builtin": "zapier", "auth_token": "user-token"}]}
```

### Multiple Services
```json
{
  "tools": [
    {"type": "mcp", "builtin": "calculator"},
    {"type": "mcp", "builtin": "zapier", "auth_token": "zapier-token"},
    {"type": "mcp", "builtin": "github", "auth_token": "github-token"}
  ]
}
```

## 🚨 Important Notes

- Ensure `examples/sample_calculator_mcp_server.py` file exists
- Python environment required
- OPENAI_API_KEY is mandatory
- External services require their respective tokens

Done! You can now easily use MCP tools! 🎉