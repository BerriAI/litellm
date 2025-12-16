import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Pydantic AI Agents

Call Pydantic AI Agents via LiteLLM's A2A Gateway.

| Property | Details |
|----------|---------|
| Description | Pydantic AI agents with native A2A support via the `to_a2a()` method. LiteLLM provides fake streaming support for agents that don't natively stream. |
| Provider Route on LiteLLM | A2A Gateway |
| Provider Doc | [Pydantic AI Agents ↗](https://ai.pydantic.dev/agents/) |

## Setup Pydantic AI Agent Server

### Overview

This example demonstrates how to create a [Pydantic AI](https://ai.pydantic.dev/agents/) agent and expose it as an A2A server using the native `to_a2a()` method.

### Install Dependencies

```bash
pip install pydantic-ai fasta2a uvicorn
```

### Create Agent

```python title="agent.py"
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini', instructions='Be helpful!')

@agent.tool_plain
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@agent.tool_plain  
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Native A2A server - Pydantic AI handles it automatically
app = agent.to_a2a()
```

### Run Server

```bash
uvicorn agent:app --host 0.0.0.0 --port 9999
```

Server runs at `http://localhost:9999`

## LiteLLM A2A Gateway

Connect to your Pydantic AI agent through LiteLLM's A2A Gateway UI.

### 1. Navigate to Agents

From the sidebar, click "Agents" to open the agent management page, then click "+ Add New Agent".

### 2. Select Pydantic AI Agent Type

Click "A2A Standard" to see available agent types, then select "Pydantic AI".

### 3. Configure the Agent

Fill in the following fields:

#### Agent Name

Enter a friendly name for your agent - callers will see this name when selecting agents.

#### Agent URL

Enter the URL where your Pydantic AI agent is running:
- Local development: `http://localhost:9999`
- Production: Your deployed agent URL

#### A2A Fields (Optional)

You can optionally configure:
- **Agent Description**: Description of what your agent does
- **Skills**: List of capabilities your agent provides

### 4. Create Agent

Click "Create Agent" to save your configuration.

### 5. Test in Playground

Go to "Playground" in the sidebar to test your agent:

1. Change the endpoint type to `/v1/a2a/message/send`
2. Select your Pydantic AI agent from the dropdown
3. Send a test message

## LiteLLM Proxy Configuration

You can also configure Pydantic AI agents directly in your `config.yaml`:

```yaml title="config.yaml"
a2a_config:
  agents:
    - agent_name: pydantic-weather-agent
      litellm_params:
        custom_llm_provider: pydantic_ai_agents
        api_base: http://localhost:9999
      agent_description: "Weather and calculator agent powered by Pydantic AI"
      skills:
        - skill_id: weather
          skill_name: Get Weather
          skill_description: Get current weather for any city
        - skill_id: calculator
          skill_name: Calculator
          skill_description: Evaluate mathematical expressions
```

Start the proxy:

```bash
litellm --config config.yaml
```

## Streaming Support

Pydantic AI agents don't natively support streaming responses. LiteLLM provides **fake streaming** - the complete response is fetched and then chunked to simulate streaming behavior.

This allows you to use streaming APIs with Pydantic AI agents:

```python
import litellm

# Fake streaming works automatically
response = await litellm.a2a_message_send(
    agent_name="pydantic-weather-agent",
    message="What's the weather in Paris?",
    stream=True,  # LiteLLM handles fake streaming
)

async for chunk in response:
    print(chunk)
```

## Further Reading

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Pydantic AI Agents](https://ai.pydantic.dev/agents/)
- [A2A Agent Gateway](../a2a.md)
- [A2A Cost Tracking](../a2a_cost_tracking.md)
