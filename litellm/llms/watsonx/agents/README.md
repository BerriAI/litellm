# IBM watsonx.ai Orchestrate Agent Support

This module provides support for IBM watsonx.ai Orchestrate Agents in LiteLLM, following the same pattern as Azure AI agents and other agent implementations.

## Model Format

```
watsonx_agent/<agent_id>
```

## API Documentation

- [Chat With Agents API](https://developer.watson-orchestrate.ibm.com/apis/orchestrate-agent/chat-with-agents)

## Usage

### Basic Example

```python
import litellm

response = litellm.completion(
    model="watsonx_agent/your-agent-id",
    messages=[
        {"role": "user", "content": "Hello, how can you help me?"}
    ],
    api_base="https://your-watsonx-api-endpoint.com",
    api_key="your-api-key"
)

print(response.choices[0].message.content)
```

### With Thread ID for Conversation Continuity

```python
import litellm

# First message - creates a new thread
response = litellm.completion(
    model="watsonx_agent/your-agent-id",
    messages=[
        {"role": "user", "content": "What's the weather like today?"}
    ],
    api_base="https://your-watsonx-api-endpoint.com",
    api_key="your-api-key"
)

# Get thread_id from response
thread_id = response._hidden_params.get("thread_id")
print(f"Thread ID: {thread_id}")

# Continue conversation with the same thread
response2 = litellm.completion(
    model="watsonx_agent/your-agent-id",
    messages=[
        {"role": "user", "content": "What about tomorrow?"}
    ],
    api_base="https://your-watsonx-api-endpoint.com",
    api_key="your-api-key",
    thread_id=thread_id  # Continue same conversation
)
```

### With Additional Parameters and Context

```python
import litellm

response = litellm.completion(
    model="watsonx_agent/your-agent-id",
    messages=[
        {"role": "user", "content": "Help me with my task"}
    ],
    api_base="https://your-watsonx-api-endpoint.com",
    api_key="your-api-key",
    additional_parameters={
        "custom_param": "value"
    },
    context={
        "user_id": "123",
        "session_info": "relevant context"
    }
)
```

### Async Example

```python
import asyncio
import litellm

async def main():
    response = await litellm.acompletion(
        model="watsonx_agent/your-agent-id",
        messages=[
            {"role": "user", "content": "Hello!"}
        ],
        api_base="https://your-watsonx-api-endpoint.com",
        api_key="your-api-key"
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

## Configuration

### Environment Variables

You can set the following environment variables:

```bash
export WATSONX_API_BASE="https://your-watsonx-api-endpoint.com"
export WATSONX_API_KEY="your-api-key"
```

Then call without explicit parameters:

```python
import litellm
import os

# Uses environment variables
response = litellm.completion(
    model="watsonx_agent/your-agent-id",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Using with LiteLLM Proxy

Add to your `config.yaml`:

```yaml
model_list:
  - model_name: my-watsonx-agent
    litellm_params:
      model: watsonx_agent/your-agent-id
      api_base: https://your-watsonx-endpoint.com
      api_key: os.environ/WATSONX_API_KEY
```

Then call via the proxy:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-proxy-key"
)

response = client.chat.completions.create(
    model="my-watsonx-agent",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## API Parameters

### Required Parameters

- `model`: Model identifier in format `watsonx_agent/{agent_id}`
- `messages`: List of message dictionaries with `role` and `content`
- `api_base`: Base URL for the watsonx API endpoint
- `api_key`: Authentication API key

### Optional Parameters

- `thread_id`: Thread ID to continue a conversation
- `additional_parameters`: Dictionary of additional parameters
- `context`: Context dictionary for the agent
- `stream`: Enable streaming (default: True)

### Response Format

The response follows the standard LiteLLM format:

```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "watsonx_agent/abc123",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response from the agent"
            },
            "finish_reason": "stop"
        }
    ],
    "_hidden_params": {
        "thread_id": "thread-xyz789"  # For conversation continuity
    }
}
```

## Authentication

The watsonx agent API supports multiple authentication methods:

1. **Bearer Token**: Directly provide a token
   ```python
   api_key="Bearer your-token"
   ```

2. **IAM API Key**: Provide an IAM API key (automatically exchanges for a bearer token)
   ```python
   api_key="your-iam-api-key"
   ```

3. **ZenApiKey**: For IBM Cloud Pak for Data environments
   ```python
   zen_api_key="your-zen-api-key"
   ```

## Error Handling

```python
import litellm
from litellm.llms.watsonx.common_utils import WatsonXAIError

try:
    response = litellm.completion(
        model="watsonx_agent/your-agent-id",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="https://your-watsonx-api-endpoint.com",
        api_key="your-api-key"
    )
except WatsonXAIError as e:
    print(f"WatsonX Error: {e.status_code} - {e.message}")
except Exception as e:
    print(f"Error: {str(e)}")
```

## Architecture

The implementation follows the same pattern as Azure AI agents:

### Files Structure

```
litellm/llms/watsonx/agents/
├── __init__.py              # Package initialization
├── transformation.py         # IBMWatsonXAgentConfig class
├── handler.py               # WatsonXAgentHandler class
└── README.md                # This file

litellm/types/llms/
└── watsonx_agents.py        # Type definitions

tests/test_litellm/llms/watsonx/agents/
└── test_watsonx_agents_transformation.py  # Tests
```

### How It Works

1. **Provider Detection**: When you use `model="watsonx_agent/agent-id"`, LiteLLM detects the `watsonx_agent` provider
2. **Static Dispatch**: Routes to `IBMWatsonXAgentConfig.completion()` static method
3. **Handler Execution**: Dispatches to `watsonx_agent_handler` singleton for sync or async execution
4. **Request Transformation**: Converts OpenAI-style messages to watsonx agent format
5. **API Call**: Makes HTTP request to watsonx Orchestrate Agent API
6. **Response Transformation**: Converts watsonx response back to OpenAI format

### Comparison with Azure AI Agents

| Aspect | Azure AI Agents | Watsonx Agents |
|--------|----------------|----------------|
| **Model Format** | `azure_ai/agents/<agent_id>` | `watsonx_agent/<agent_id>` |
| **API Flow** | Multi-step (thread → messages → run → poll) | Single API call |
| **Threading** | Manual thread management | Thread ID in header |
| **Authentication** | Azure AD Bearer tokens | IAM/Bearer tokens |
| **Static Dispatch** | ✅ Yes | ✅ Yes |
| **Async Support** | ✅ Yes | ✅ Yes |

## Features

- ✅ Synchronous and asynchronous completions
- ✅ Thread-based conversation management
- ✅ Multiple authentication methods
- ✅ Custom parameters and context support
- ✅ Error handling with detailed error messages
- ✅ Follows LiteLLM agent implementation patterns
- ✅ Compatible with LiteLLM Proxy
- ✅ Environment variable configuration

## Testing

Run the tests with:

```bash
poetry run pytest tests/test_litellm/llms/watsonx/agents/ -v
```

## Implementation Details

### IBMWatsonXAgentConfig

Configuration class that handles:
- Parameter mapping
- URL building
- Request/response transformation
- Environment validation
- Static dispatch to handler

### WatsonXAgentHandler

Handler class that executes:
- Synchronous completions
- Asynchronous completions
- HTTP client management
- Error handling

## Contributing

When contributing to watsonx agent support:

1. Follow the existing code patterns in the watsonx module
2. Match the Azure AI agents implementation pattern
3. Add tests for new features
4. Update this README with new examples
5. Ensure all tests pass with `make test-unit`

## Comparison with Other LiteLLM Agent Implementations

### Azure AI Agents
- Multi-step flow with polling
- Thread and run management
- Similar static dispatch pattern

### Langraph Agents
- Graph-based execution
- State management
- Different execution model

### Watsonx Agents (This Implementation)
- Single API call model
- Simple thread continuation
- Follows Azure AI pattern for consistency
