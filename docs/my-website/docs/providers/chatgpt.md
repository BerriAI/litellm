# ChatGPT Subscription

Use ChatGPT Pro/Max subscription models through LiteLLM with OAuth device flow authentication.

| Property | Details |
|-------|-------|
| Description | ChatGPT subscription access (Codex + GPT-5.2 family) via ChatGPT backend API |
| Provider Route on LiteLLM | `chatgpt/` |
| Supported Endpoints | `/responses`, `/chat/completions` (bridged to Responses for supported models) |
| API Reference | https://chatgpt.com |

ChatGPT subscription access is native to the Responses API. Chat Completions requests are bridged to Responses for supported models (for example `chatgpt/gpt-5.2`).

Notes:
- The ChatGPT subscription backend rejects token limit fields (`max_tokens`, `max_output_tokens`, `max_completion_tokens`) and `metadata`. LiteLLM strips these fields for this provider.
- `/v1/chat/completions` honors `stream`. When `stream` is false (default), LiteLLM aggregates the Responses stream into a single JSON response.

## Authentication

ChatGPT subscription access uses an OAuth device code flow:

1. LiteLLM prints a device code and verification URL
2. Open the URL, sign in, and enter the code
3. Tokens are stored locally for reuse

## Usage - LiteLLM Python SDK

### Responses (recommended for Codex models)

```python showLineNumbers title="ChatGPT Responses"
import litellm

response = litellm.responses(
    model="chatgpt/gpt-5.2-codex",
    input="Write a Python hello world"
)

print(response)
```

### Chat Completions (bridged to Responses)

```python showLineNumbers title="ChatGPT Chat Completions"
import litellm

response = litellm.completion(
    model="chatgpt/gpt-5.2",
    messages=[{"role": "user", "content": "Write a Python hello world"}]
)

print(response)
```

## Usage - LiteLLM Proxy

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: chatgpt/gpt-5.2
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.2
  - model_name: chatgpt/gpt-5.2-codex
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.2-codex
```

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml
```

## Configuration

### Environment Variables

- `CHATGPT_TOKEN_DIR`: Custom token storage directory
- `CHATGPT_AUTH_FILE`: Auth file name (default: `auth.json`)
- `CHATGPT_API_BASE`: Override API base (default: `https://chatgpt.com/backend-api/codex`)
- `OPENAI_CHATGPT_API_BASE`: Alias for `CHATGPT_API_BASE`
- `CHATGPT_ORIGINATOR`: Override the `originator` header value
- `CHATGPT_USER_AGENT`: Override the `User-Agent` header value
- `CHATGPT_USER_AGENT_SUFFIX`: Optional suffix appended to the `User-Agent` header
