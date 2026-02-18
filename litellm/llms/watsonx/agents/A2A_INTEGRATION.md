# WatsonX Agents with A2A Protocol (Agent Gateway)

WatsonX agents can be exposed through LiteLLM's A2A Agent Gateway using the **completion bridge** adapter. This allows you to invoke WatsonX agents using the A2A JSON-RPC protocol while maintaining all LiteLLM features like logging, cost tracking, and access control.

## How It Works

The completion bridge acts as an adapter that:

1. **Receives A2A JSON-RPC requests** (with messages in A2A format)
2. **Transforms to OpenAI format** using transformation utilities
3. **Routes through `litellm.completion()`** with `watsonx_agent` provider
4. **Converts responses back to A2A JSON-RPC format**

This means WatsonX agents work with the A2A protocol **without requiring native A2A support** in the WatsonX API itself.

## Architecture

```
A2A Client → LiteLLM A2A Endpoint → Completion Bridge → WatsonX Agent Handler → WatsonX API
   (A2A)         (A2A ↔ OpenAI)        (OpenAI ↔ WatsonX)
```

## Configuration

### Option 1: LiteLLM Proxy Configuration

Add your WatsonX agent to `proxy_server_config.yaml`:

```yaml
model_list:
  - model_name: my-watsonx-agent
    litellm_params:
      model: watsonx_agent/your-agent-id
      api_base: https://your-watsonx-endpoint.com
      api_key: os.environ/WATSONX_API_KEY

# Register agent for A2A access
agents:
  - agent_id: watsonx-assistant
    agent_name: WatsonX Assistant
    litellm_params:
      custom_llm_provider: watsonx
      model: watsonx_agent/your-agent-id
      api_key: os.environ/WATSONX_API_KEY
    agent_card_params:
      name: WatsonX Assistant
      description: AI assistant powered by IBM WatsonX
      url: https://your-watsonx-endpoint.com
```

Start the proxy:

```bash
litellm --config proxy_server_config.yaml
```

### Option 2: Direct Python SDK Usage

```python
from litellm.a2a_protocol import asend_message
from a2a.types import SendMessageRequest, MessageSendParams
from uuid import uuid4

# Create A2A request
request = SendMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello, how can you help me?"}],
            "messageId": uuid4().hex,
        }
    ),
)

# Send via completion bridge
response = await asend_message(
    request=request,
    api_base="https://your-watsonx-endpoint.com",
    litellm_params={
        "custom_llm_provider": "watsonx",
        "model": "watsonx_agent/your-agent-id",
        "api_key": "your-api-key",
    },
)

print(response.result["message"]["parts"][0]["text"])
```

## Invoking the Agent

### Using A2A SDK

```python
from a2a.client import A2AClient
from a2a.types import SendMessageRequest, MessageSendParams
from uuid import uuid4

# Connect to LiteLLM proxy
client = await A2AClient.create(base_url="http://localhost:4000/a2a/watsonx-assistant")

# Send message
request = SendMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={
            "role": "user",
            "parts": [{"kind": "text", "text": "What can you do?"}],
            "messageId": uuid4().hex,
        }
    ),
)

response = await client.send_message(request)
print(response.result.message.parts[0].text)
```

### Using OpenAI SDK (Alternative)

You can also invoke via the standard OpenAI-compatible `/chat/completions` endpoint:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-proxy-key"
)

response = client.chat.completions.create(
    model="a2a/watsonx-assistant",  # Note: a2a/ prefix
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### Using HTTP/cURL

```bash
# A2A Protocol endpoint
curl -X POST http://localhost:4000/a2a/watsonx-assistant/message/send \
  -H "Authorization: Bearer sk-your-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "req-123",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello!"}],
        "messageId": "msg-123"
      }
    }
  }'
```

## Streaming Support

WatsonX agents support streaming through the A2A protocol:

```python
from litellm.a2a_protocol import asend_message_streaming
from a2a.types import SendStreamingMessageRequest, MessageSendParams
from uuid import uuid4

request = SendStreamingMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={
            "role": "user",
            "parts": [{"kind": "text", "text": "Tell me a story"}],
            "messageId": uuid4().hex,
        }
    ),
)

async for chunk in asend_message_streaming(
    request=request,
    api_base="https://your-watsonx-endpoint.com",
    litellm_params={
        "custom_llm_provider": "watsonx",
        "model": "watsonx_agent/your-agent-id",
        "api_key": "your-api-key",
    },
):
    print(chunk)
```

The streaming response follows the A2A protocol with these events:
1. **Task event** (`kind: "task"`) - Initial task creation
2. **Status update** (`kind: "status-update"`) - Status change to "working"
3. **Artifact update** (`kind: "artifact-update"`) - Content delivery
4. **Status update** (`kind: "status-update"`) - Final "completed" status

## Thread Continuity

WatsonX agents support conversation continuity through thread IDs:

```python
# First message - creates a new thread
response1 = await asend_message(
    request=request1,
    api_base=api_base,
    litellm_params={
        "custom_llm_provider": "watsonx",
        "model": "watsonx_agent/your-agent-id",
        "api_key": api_key,
    },
)

# Get thread_id from response
thread_id = response1._hidden_params.get("thread_id")

# Continue conversation with same thread
response2 = await asend_message(
    request=request2,
    api_base=api_base,
    litellm_params={
        "custom_llm_provider": "watsonx",
        "model": "watsonx_agent/your-agent-id",
        "api_key": api_key,
        "thread_id": thread_id,  # Continue same conversation
    },
)
```

## Features

| Feature | Supported |
|---------|-----------|
| **A2A Protocol** | ✅ (via completion bridge) |
| **Streaming** | ✅ |
| **Thread Continuity** | ✅ |
| **Cost Tracking** | ✅ |
| **Logging** | ✅ |
| **Access Control** | ✅ |
| **Load Balancing** | ✅ |

## Benefits of A2A Integration

1. **Standardized Protocol**: Use the same A2A protocol across different agent providers (WatsonX, Vertex AI, Azure AI, LangGraph, etc.)
2. **Cost Tracking**: Automatic cost calculation and logging for WatsonX agent usage
3. **Access Control**: Team-based and key-based access control to agents
4. **Observability**: Unified logging across all agent calls through LiteLLM
5. **Discovery**: Agents are discoverable through the AI Hub registry

## Comparison with Other Providers

| Provider | A2A Support | Implementation |
|----------|-------------|----------------|
| Native A2A Agents | ✅ | Direct A2A protocol |
| Vertex AI Agent Engine | ✅ | Completion bridge |
| Azure AI Foundry | ✅ | Completion bridge |
| LangGraph | ✅ | Completion bridge |
| Bedrock AgentCore | ✅ | Completion bridge |
| **WatsonX Agents** | ✅ | **Completion bridge** |

## Authentication

WatsonX agents support multiple authentication methods:

1. **Bearer Token**:
   ```python
   litellm_params={"api_key": "Bearer your-token"}
   ```

2. **IAM API Key** (automatically exchanges for token):
   ```python
   litellm_params={"api_key": "your-iam-api-key"}
   ```

3. **ZenApiKey** (for IBM Cloud Pak for Data):
   ```python
   litellm_params={"zen_api_key": "your-zen-api-key"}
   ```

## Example: Full Integration

Here's a complete example using WatsonX agents with the LiteLLM proxy:

```yaml
# proxy_server_config.yaml
model_list:
  - model_name: customer-support-agent
    litellm_params:
      model: watsonx_agent/agent-123
      api_base: https://watsonx.example.com
      api_key: os.environ/WATSONX_API_KEY

agents:
  - agent_id: customer-support
    agent_name: Customer Support Agent
    litellm_params:
      custom_llm_provider: watsonx
      model: watsonx_agent/agent-123
      api_key: os.environ/WATSONX_API_KEY
    agent_card_params:
      name: Customer Support Agent
      description: Handles customer inquiries and support tickets
      url: https://watsonx.example.com
      capabilities:
        - text_generation
        - conversation
```

Client code:

```python
from a2a.client import A2AClient

# Connect to agent via LiteLLM proxy
client = await A2AClient.create(
    base_url="http://localhost:4000/a2a/customer-support",
    headers={"Authorization": "Bearer sk-your-litellm-key"}
)

# Use the agent
response = await client.send_message(request)
```

## Troubleshooting

### Agent not found
- Ensure the agent is registered in `agents:` section of config
- Check that `agent_id` matches the URL path: `/a2a/{agent_id}`

### Authentication errors
- Verify `WATSONX_API_KEY` is set correctly
- Check that the API key has access to the specified agent
- For IAM keys, ensure they're not expired

### Response format issues
- The completion bridge automatically handles format conversion
- Check LiteLLM logs for transformation errors: `litellm --debug`

## Testing

Run the test suite:

```bash
# Set environment variables
export WATSONX_API_BASE="https://your-endpoint.com"
export WATSONX_API_KEY="your-api-key"
export WATSONX_AGENT_ID="your-agent-id"

# Run tests
pytest tests/agent_tests/local_only_agent_tests/test_a2a_watsonx_agent.py -v -s
```

## References

- [LiteLLM Agent Gateway Documentation](https://docs.litellm.ai/docs/a2a)
- [WatsonX Agents Documentation](./README.md)
- [A2A Protocol Specification](https://github.com/google/a2a)
