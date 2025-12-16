import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Pydantic AI Agents

Use Pydantic AI Agents on LiteLLM AI Gateway 

| Property | Details |
|----------|---------|
| Description | Azure AI Foundry Agents provides hosted agent runtimes that can execute agentic workflows with foundation models, tools, and code interpreters. |
| Provider Route on LiteLLM | `azure_ai/agents/{AGENT_ID}` |
| Provider Doc | [Azure AI Foundry Agents ↗](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart) |

## LiteLLM A2A GTateway usage 
## Setup Pydantic AI Agent Server

## Overview

This will start the pydantic ai agent on localhost:9999

This example demonstrates how to create a [Pydantic AI](https://ai.pydantic.dev/agents/) agent and expose it as an A2A server using the native `to_a2a()` method. The agent includes:

- **Weather tool**: Get mock weather data for cities
- **Calculator tool**: Perform mathematical calculations

## Setup

```bash
cd samples/python/agents/pydantic_ai_local
echo "OPENAI_API_KEY=your_key" > .env
pip install -e .
```

## Run

```bash
uvicorn agent:app --host 0.0.0.0 --port 9999
```

Server runs at `http://localhost:9999`

## Test

```bash
python test_client.py
```

## How It Works

Pydantic AI has native A2A support via the `to_a2a()` method:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini', instructions='Be helpful!')

@agent.tool_plain
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny"

# That's it! Native A2A server
app = agent.to_a2a()
```


No manual A2A wrapping needed - Pydantic AI handles it nativ

## Add Agent on LiteLLM


