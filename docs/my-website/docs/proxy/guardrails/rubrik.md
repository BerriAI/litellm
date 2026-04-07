# Rubrik Guardrail

Use Rubrik's tool blocking and logging integration to validate LLM tool calls against an external policy service and batch-log all LLM requests/responses.

**Key features:**
- **Tool blocking**: Validates tool calls against an external Rubrik service after LLM completion. Blocked tool calls trigger a policy violation response.
- **Batch logging**: Logs all LLM requests and responses to Rubrik with configurable sampling and batching.
- **Fail-open**: If the tool blocking service is unavailable, requests are allowed through unchanged.

---

## Quick Start

### 1. Set Environment Variables

```bash
export RUBRIK_WEBHOOK_URL="https://your-rubrik-service.example.com"
export RUBRIK_API_KEY="your-rubrik-api-key"        # optional
```

### 2. Configure `config.yaml`

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "rubrik"
    litellm_params:
      guardrail: rubrik
      mode: "post_call"
      api_key: os.environ/RUBRIK_API_KEY
      api_base: os.environ/RUBRIK_WEBHOOK_URL
      default_on: true
```

### 3. Launch the Proxy

```bash
litellm --config config.yaml --port 4000
```

### 4. Test It

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "What is the weather in SF?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RUBRIK_WEBHOOK_URL` | Yes (or `api_base` in config) | — | Base URL of the Rubrik webhook service |
| `RUBRIK_API_KEY` | No | — | Bearer token for authenticating with the Rubrik service |
| `RUBRIK_SAMPLING_RATE` | No | `1.0` | Fraction of requests to log (0.0 to 1.0). Set to `0.5` to log ~50% of requests. |
| `RUBRIK_BATCH_SIZE` | No | `512` | Number of log entries to buffer before flushing. Logs are also flushed on a periodic interval. |

### YAML Config Parameters

| Parameter | Description |
|-----------|-------------|
| `guardrail: rubrik` | Selects the Rubrik guardrail integration |
| `mode: "post_call"` | Run after the LLM response is received |
| `api_key` | Rubrik API key (can use `os.environ/RUBRIK_API_KEY`) |
| `api_base` | Rubrik webhook base URL (can use `os.environ/RUBRIK_WEBHOOK_URL`) |
| `default_on` | When `true`, the guardrail runs on all requests without needing per-request opt-in |

---

## How Tool Blocking Works

1. After the LLM returns a response with tool calls, the Rubrik guardrail sends them to the blocking service at `{api_base}/v1/after_completion/openai/v1`.
2. The service evaluates each tool call against configured policies and returns the set of **allowed** tool calls.
3. If any tool calls are blocked, the proxy returns a `200` response with the policy violation explanation instead of the original LLM response.
4. If the blocking service is unreachable or returns an error, the guardrail **fails open** — the original response is returned unchanged.

### Request/Response format

The guardrail sends a JSON envelope to the blocking service:

```json
{
  "request": {
    "messages": [...],
    "model": "gpt-4",
    "proxy_server_request": {...}
  },
  "response": {
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "choices": [{
      "message": {
        "role": "assistant",
        "tool_calls": [...]
      }
    }]
  }
}
```

The service should return an OpenAI chat completion format response containing only the **allowed** tool calls and an optional `content` field with the blocking explanation.

---

## How Batch Logging Works

All LLM requests (successes and failures) are queued and sent in batches to `{api_base}/v1/litellm/batch`.

- Logs are flushed when the queue reaches `RUBRIK_BATCH_SIZE` (default 512) or on a periodic interval (default 5 seconds).
- Use `RUBRIK_SAMPLING_RATE` to reduce logging volume in high-traffic deployments.
- For Anthropic `/v1/messages` requests, the log ID is normalized to `litellm_call_id` for consistency across tool blocking and logging.
