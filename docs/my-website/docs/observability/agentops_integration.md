# 🖇️ AgentOps - LLM Observability Platform

:::tip

This is community maintained. Please make an issue if you run into a bug:
https://github.com/BerriAI/litellm

:::

[AgentOps](https://docs.agentops.ai) is an observability platform that enables tracing and monitoring of LLM calls, providing detailed insights into your AI operations.

## Using AgentOps with LiteLLM

LiteLLM provides `success_callbacks` and `failure_callbacks`, allowing you to easily integrate AgentOps for comprehensive tracing and monitoring of your LLM operations.

### Integration

Use just a few lines of code to instantly trace your responses **across all providers** with AgentOps:

```python
from agentops.integration.callbacks.litellm import LiteLLMCallbackHandler
import litellm

# Initialize the callback handler
callback_handler = LiteLLMCallbackHandler(
    api_key="your_agentops_api_key",  # Optional
    tags=["production", "chatbot"],    # Optional
)

# Configure LiteLLM to use the callback
litellm.callbacks = [callback_handler]
```

Complete Code:

```python
import os
from litellm import completion
from agentops.integration.callbacks.litellm import LiteLLMCallbackHandler

# Set env variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Initialize AgentOps callback
callback_handler = LiteLLMCallbackHandler(
    api_key="your_agentops_api_key",
    tags=["production"]
)

# Configure LiteLLM
litellm.callbacks = [callback_handler]

# OpenAI call
response = completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hi 👋 - I'm OpenAI"}],
)

print(response)
```

### Features

The AgentOps integration provides comprehensive tracing and monitoring capabilities:

- **Automatic Session Management**
  - Automatic session span creation and management
  - Hierarchical tracing of multi-step operations

- **Detailed Tracing**
  - Pre-request tracing
  - Success event tracking
  - Failure event monitoring
  - Support for both synchronous and asynchronous operations

- **Rich Span Attributes**
  - Provider information
  - Model details
  - Request parameters (temperature, max_tokens, streaming)
  - Message history
  - Response data
  - Timing information
  - Error details (when applicable)

### Configuration Options

The `LiteLLMCallbackHandler` accepts the following parameters:

- `api_key` (str, optional): Your AgentOps API key
- `tags` (List[str], optional): Tags to add to the session for better organization

### Advanced Usage

You can add custom tags and metadata to your traces:

```python
callback_handler = LiteLLMCallbackHandler(
    api_key="your_agentops_api_key",
    tags=["production", "chatbot", "customer-service"],
)
```

### Support

For issues or questions, please refer to:
- [AgentOps Documentation](https://docs.agentops.ai)
- [LiteLLM Documentation](https://docs.litellm.ai) 