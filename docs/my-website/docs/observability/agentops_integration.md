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
Get your AgentOps API Keys from https://app.agentops.ai/
```python
import litellm

# Configure LiteLLM to use AgentOps
litellm.success_callback = ["agentops"]

# Make your LLM calls as usual
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)
```

Complete Code:

```python
import os
from litellm import completion

# Set env variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["AGENTOPS_API_KEY"] = "your-agentops-api-key"

# Configure LiteLLM to use AgentOps
litellm.success_callback = ["agentops"]

# OpenAI call
response = completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hi 👋 - I'm OpenAI"}],
)

print(response)
```

### Configuration Options

The AgentOps integration can be configured through environment variables:

- `AGENTOPS_API_KEY` (str, optional): Your AgentOps API key
- `AGENTOPS_ENVIRONMENT` (str, optional): Deployment environment (defaults to "production")
- `AGENTOPS_SERVICE_NAME` (str, optional): Service name for tracing (defaults to "agentops")

### Advanced Usage

You can configure additional settings through environment variables:

```python
import os

# Configure AgentOps settings
os.environ["AGENTOPS_API_KEY"] = "your-agentops-api-key"
os.environ["AGENTOPS_ENVIRONMENT"] = "staging"
os.environ["AGENTOPS_SERVICE_NAME"] = "my-service"

# Enable AgentOps tracing
litellm.success_callback = ["agentops"]
```

### Support

For issues or questions, please refer to:
- [AgentOps Documentation](https://docs.agentops.ai)
- [LiteLLM Documentation](https://docs.litellm.ai) 