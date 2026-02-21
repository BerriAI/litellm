import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenClaw

[OpenClaw](https://openclaw.ai) is an AI agent framework that exposes an OpenAI-compatible HTTP endpoint. It allows you to interact with AI agents that have access to tools, memory, and custom configurations.

## Key Features

- **Agent Targeting**: Route requests to specific agents via the model field (`openclaw/main`, `openclaw/research`)
- **Session Persistence**: Maintain conversation context across requests using the `user` field
- **Full Tool Access**: Agents can execute code, browse the web, manage files, and more
- **Streaming Support**: Real-time SSE streaming responses

## Quick Start

```python
import litellm

response = litellm.completion(
    model="openclaw/main",  # Target the 'main' agent
    api_base="http://localhost:18789",
    api_key="your-gateway-token",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Environment Variables

```bash
export OPENCLAW_API_BASE="http://localhost:18789"  # Gateway URL
export OPENCLAW_API_KEY="your-gateway-token"       # Auth token
```

## Usage

### SDK Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm

# Basic completion
response = litellm.completion(
    model="openclaw/main",
    messages=[{"role": "user", "content": "What can you do?"}]
)

# Streaming
response = litellm.completion(
    model="openclaw/main",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")

# With session persistence (same user = same conversation)
response = litellm.completion(
    model="openclaw/main",
    messages=[{"role": "user", "content": "Remember my name is Alice"}],
    user="alice-session-123"
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Add to your `config.yaml`:

```yaml
model_list:
  - model_name: openclaw-main
    litellm_params:
      model: openclaw/main
      api_base: http://localhost:18789
      api_key: os.environ/OPENCLAW_API_KEY
  
  - model_name: openclaw-research
    litellm_params:
      model: openclaw/research
      api_base: http://localhost:18789
      api_key: os.environ/OPENCLAW_API_KEY
```

2. Start the proxy:

```bash
litellm --config config.yaml
```

3. Make requests:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "openclaw-main",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

</TabItem>
</Tabs>

## Targeting Different Agents

OpenClaw can run multiple agents with different configurations. Target them via the model field:

```python
# Main agent (default)
litellm.completion(model="openclaw/main", ...)

# Research agent with web search tools
litellm.completion(model="openclaw/research", ...)

# Custom agent
litellm.completion(model="openclaw/my-custom-agent", ...)
```

## Supported Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Agent to target: `openclaw/<agent-id>` |
| `messages` | array | Conversation messages |
| `stream` | boolean | Enable SSE streaming |
| `temperature` | float | Sampling temperature |
| `max_tokens` | integer | Maximum tokens to generate |
| `user` | string | Session key for conversation persistence |
| `tools` | array | Tool definitions (agents have built-in tools) |
| `tool_choice` | string/object | Tool selection preference |

## OpenClaw Setup

To use OpenClaw with LiteLLM:

1. Install and start OpenClaw:
```bash
npm install -g openclaw
openclaw gateway
```

2. Enable the HTTP endpoint in your OpenClaw config:
```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    }
  }
}
```

3. Configure authentication (optional but recommended):
```bash
export OPENCLAW_GATEWAY_TOKEN="your-secure-token"
```

For more details, see the [OpenClaw documentation](https://docs.openclaw.ai).

## Troubleshooting

### Connection Refused
Ensure the OpenClaw gateway is running and the API base URL is correct:
```bash
curl http://localhost:18789/health
```

### Authentication Failed
Verify your gateway token matches the one configured in OpenClaw:
```bash
openclaw gateway status
```

### Agent Not Found
Check available agents:
```bash
openclaw agents list
```
