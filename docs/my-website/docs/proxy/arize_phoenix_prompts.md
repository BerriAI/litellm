# Arize Phoenix Prompt Management

Use prompt versions from [Arize Phoenix](https://phoenix.arize.com/) with LiteLLM SDK and Proxy.

## Quick Start

### SDK

```python
import litellm

response = litellm.completion(
    model="gpt-4o",
    prompt_id="UHJvbXB0VmVyc2lvbjox",
    prompt_integration="arize_phoenix",
    api_key="your-arize-phoenix-token",
    api_base="https://app.phoenix.arize.com/s/your-workspace",
    prompt_variables={"question": "What is AI?"},
)
```

### Proxy

**1. Add prompt to config**

```yaml
prompts:
  - prompt_id: "simple_prompt"
    litellm_params:
      prompt_id: "UHJvbXB0VmVyc2lvbjox"
      prompt_integration: "arize_phoenix"
      api_base: https://app.phoenix.arize.com/s/your-workspace
      api_key: os.environ/PHOENIX_API_KEY
      ignore_prompt_manager_model: true # optional: use model from config instead
      ignore_prompt_manager_optional_params: true # optional: ignore temp, max_tokens from prompt
```

**2. Make request**

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-3.5-turbo",
    "prompt_id": "simple_prompt",
    "prompt_variables": {
      "question": "Explain quantum computing"
    }
  }'
```

## Configuration

### Get Arize Phoenix Credentials

1. **API Token**: Get from [Arize Phoenix Settings](https://app.phoenix.arize.com/)
2. **Workspace URL**: `https://app.phoenix.arize.com/s/{your-workspace}`
3. **Prompt ID**: Found in prompt version URL

**Set environment variable**:
```bash
export PHOENIX_API_KEY="your-token"
```

### SDK + PROXY Options

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt_id` | Yes | Arize Phoenix prompt version ID |
| `prompt_integration` | Yes | Set to `"arize_phoenix"` |
| `api_base` | Yes | Workspace URL |
| `api_key` | Yes | Access token |
| `prompt_variables` | No | Variables for template |

### Proxy-only Options

| Parameter | Description |
|-----------|-------------|
| `ignore_prompt_manager_model` | Use config model instead of prompt's model |
| `ignore_prompt_manager_optional_params` | Ignore temperature, max_tokens from prompt |

## Variable Templates

Arize Phoenix uses Mustache/Handlebars syntax:

```python
# Template: "Hello {{name}}, question: {{question}}"
prompt_variables = {
    "name": "Alice",
    "question": "What is ML?"
}
# Result: "Hello Alice, question: What is ML?"
```


## Combine with Additional Messages

```python
response = litellm.completion(
    model="gpt-4o",
    prompt_id="UHJvbXB0VmVyc2lvbjox",
    prompt_integration="arize_phoenix",
    api_base="https://app.phoenix.arize.com/s/your-workspace",
    prompt_variables={"question": "Explain AI"},
    messages=[
        {"role": "user", "content": "Keep it under 50 words"}
    ]
)
```


## Error Handling

```python
try:
    response = litellm.completion(
        model="gpt-4o",
        prompt_id="invalid-id",
        prompt_integration="arize_phoenix",
        api_base="https://app.phoenix.arize.com/s/workspace"
    )
except Exception as e:
    print(f"Error: {e}")
    # 404: Prompt not found
    # 401: Invalid credentials
    # 403: Access denied
```

## Support

- [LiteLLM GitHub Issues](https://github.com/BerriAI/litellm/issues)
- [Arize Phoenix Docs](https://docs.arize.com/phoenix)

