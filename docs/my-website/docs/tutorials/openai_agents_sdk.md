import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI Agents SDK with LiteLLM

Use OpenAI's Agents SDK with any LLM provider through LiteLLM Proxy.

This tutorial shows you how to build AI agents using the OpenAI Agents SDK with support for multiple LLM providers through LiteLLM.

## Overview

The OpenAI Agents SDK provides a high-level interface for building AI agents. By integrating with LiteLLM, you can:

- Use multiple LLM providers (Bedrock, Azure, Vertex AI, etc.) with the same agent code
- Switch easily between models from different providers
- Connect to a LiteLLM proxy for centralized model management

:::tip Built-in LiteLLM Extension

The OpenAI Agents SDK includes an official LiteLLM extension (`LitellmModel`) that works without a proxy. If you don't need centralized proxy features (cost tracking, rate limiting, load balancing), you can use it directly:

```python
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel


agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=LitellmModel(model="anthropic/claude-sonnet-4-20250514"),
)

result = Runner.run_sync(agent, "Hello!")
print(result.final_output)
```

See the [Docs](https://openai.github.io/openai-agents-python/models/litellm/) for more details. The rest of this tutorial focuses on the **proxy-based approach** for teams that need centralized model management.

:::

## Prerequisites

- Python environment setup
- API keys for your LLM providers
- Basic understanding of LLMs and agent concepts

## Installation

```bash showLineNumbers title="Install dependencies"
pip install openai-agents litellm
```

## 1. Start LiteLLM Proxy

Configure and start the LiteLLM proxy with the models you want to use:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: bedrock-claude-sonnet-4
    litellm_params:
      model: "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
      aws_region_name: "us-east-1"

  - model_name: gpt-4o
    litellm_params:
      model: "openai/gpt-4o"

  - model_name: claude-sonnet-4
    litellm_params:
      model: "anthropic/claude-sonnet-4-20250514"

  - model_name: bedrock-claude-haiku
    litellm_params:
      model: "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"
      aws_region_name: "us-east-1"

  - model_name: bedrock-nova-premier
    litellm_params:
      model: "bedrock/amazon.nova-premier-v1:0"
      aws_region_name: "us-east-1"
```

```bash
litellm --config config.yaml
```

Required environment variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `LITELLM_API_KEY` | `sk-1234` | Your LiteLLM API key (not your provider's key) |

## 2. Setting Up Environment

Import the necessary libraries and configure your LiteLLM proxy connection:

```python showLineNumbers title="Setup environment"
from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_disabled,
)

# Point to LiteLLM proxy
BASE_URL = os.getenv("LITELLM_BASE_URL") or "http://localhost:4000"
API_KEY = os.getenv("LITELLM_API_KEY") or "sk-1234"

# Define model constants for cleaner code
MODEL_BEDROCK_SONNET = "bedrock-claude-sonnet-4"
MODEL_BEDROCK_HAIKU = "bedrock-claude-haiku"
MODEL_GPT_4O = "gpt-4o"

# Create the OpenAI client pointed at LiteLLM
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# Disable tracing since we're not using OpenAI's platform directly
set_tracing_disabled(disabled=True)
```

## 3. Create a Custom Model Provider

The Agents SDK uses a `ModelProvider` to resolve model names. Create a custom provider that routes all requests through LiteLLM:

```python showLineNumbers title="Custom LiteLLM model provider"
class LiteLLMModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(
            model=model_name or MODEL_BEDROCK_SONNET,
            openai_client=client,
        )


LITELLM_MODEL_PROVIDER = LiteLLMModelProvider()
```

## 4. Define a Simple Tool

Create a tool that your agent can use:

```python showLineNumbers title="Weather tool implementation"
@function_tool
def get_weather(city: str) -> str:
    """Retrieves the current weather report for a specified city.

    Args:
        city: The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        A string containing the weather information for the city.
    """
    print(f"[debug] getting weather for {city}")

    mock_weather_db = {
        "new york": "The weather in New York is sunny with a temperature of 25°C.",
        "london": "It's cloudy in London with a temperature of 15°C.",
        "tokyo": "Tokyo is experiencing light rain and a temperature of 18°C.",
    }

    city_normalized = city.lower()

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return f"Sorry, I don't have weather information for '{city}'."
```

## 5. Using Different Models with Agents

### 5.1 Using Bedrock Models

```python showLineNumbers title="Bedrock model via LiteLLM proxy"
async def test_bedrock_agent():
    print("\n--- Testing Bedrock Claude Agent ---")

    agent = Agent(
        name="weather_agent_bedrock",
        instructions="You are a helpful weather assistant powered by Claude. "
                     "Use the 'get_weather' tool for city weather requests. "
                     "Present information clearly.",
        tools=[get_weather],
    )

    result = await Runner.run(
        agent,
        "What's the weather in Tokyo?",
        run_config=RunConfig(
            model_provider=LITELLM_MODEL_PROVIDER,
            model="bedrock-claude-sonnet-4",  # Uses the model name from your LiteLLM config
        ),
    )
    print(f"<<< Agent Response: {result.final_output}")


asyncio.run(test_bedrock_agent())
```

### 5.2 Using OpenAI Models

```python showLineNumbers title="OpenAI model via LiteLLM proxy"
async def test_openai_agent():
    print("\n--- Testing OpenAI GPT Agent ---")

    agent = Agent(
        name="weather_agent_gpt",
        instructions="You are a helpful weather assistant powered by GPT-4o. "
                     "Use the 'get_weather' tool for city weather requests. "
                     "Present information clearly.",
        tools=[get_weather],
    )

    result = await Runner.run(
        agent,
        "What's the weather in London?",
        run_config=RunConfig(
            model_provider=LITELLM_MODEL_PROVIDER,
            model="gpt-4o",  # Uses the model name from your LiteLLM config
        ),
    )
    print(f"<<< Agent Response: {result.final_output}")


asyncio.run(test_openai_agent())
```

### 5.3 Using Anthropic Models

```python showLineNumbers title="Anthropic model via LiteLLM proxy"
async def test_anthropic_agent():
    print("\n--- Testing Anthropic Claude Agent ---")

    agent = Agent(
        name="weather_agent_claude",
        instructions="You are a helpful weather assistant powered by Claude. "
                     "Use the 'get_weather' tool for city weather requests. "
                     "Present information clearly.",
        tools=[get_weather],
    )

    result = await Runner.run(
        agent,
        "What's the weather in New York?",
        run_config=RunConfig(
            model_provider=LITELLM_MODEL_PROVIDER,
            model="claude-sonnet-4",  # Uses the model name from your LiteLLM config
        ),
    )
    print(f"<<< Agent Response: {result.final_output}")


asyncio.run(test_anthropic_agent())
```

## 6. Complete Working Example

Here's a full end-to-end script you can copy and run:

```python showLineNumbers title="complete_agent.py"
from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_disabled,
)

# Point to LiteLLM proxy
BASE_URL = os.getenv("LITELLM_BASE_URL") or "http://localhost:4000"
API_KEY = os.getenv("LITELLM_API_KEY") or "sk-1234"
MODEL_NAME = os.getenv("MODEL_NAME") or "bedrock-claude-sonnet-4"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)


class LiteLLMModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(
            model=model_name or MODEL_NAME,
            openai_client=client,
        )


LITELLM_MODEL_PROVIDER = LiteLLMModelProvider()


@function_tool
def get_weather(city: str) -> str:
    """Retrieves the current weather report for a specified city."""
    print(f"[debug] getting weather for {city}")

    mock_weather_db = {
        "new york": "The weather in New York is sunny with a temperature of 25°C.",
        "london": "It's cloudy in London with a temperature of 15°C.",
        "tokyo": "Tokyo is experiencing light rain and a temperature of 18°C.",
    }

    city_normalized = city.lower()
    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return f"Sorry, I don't have weather information for '{city}'."


async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful weather assistant. "
                     "Use the 'get_weather' tool for city weather requests. "
                     "Present information clearly and concisely.",
        tools=[get_weather],
    )

    # Run with the default model (bedrock-claude-sonnet-4)
    result = await Runner.run(
        agent,
        "What's the weather in Tokyo?",
        run_config=RunConfig(model_provider=LITELLM_MODEL_PROVIDER),
    )
    print(result.final_output)

    # Switch to a different model by passing model in RunConfig
    result = await Runner.run(
        agent,
        "What's the weather in London?",
        run_config=RunConfig(
            model_provider=LITELLM_MODEL_PROVIDER,
            model="gpt-4o",
        ),
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

## Why Use LiteLLM with Agents SDK?

| Feature | Benefit |
|---------|---------|
| **Multi-Provider** | Use the same agent code with OpenAI, Bedrock, Azure, Vertex AI, etc. |
| **Cost Tracking** | Track spending across all agent conversations |
| **Rate Limiting** | Set budgets and limits on agent usage |
| **Load Balancing** | Distribute requests across multiple API keys or regions |
| **Fallbacks** | Automatically retry with different models if one fails |

## Related Resources

- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/)
- [LiteLLM Proxy Quick Start](../proxy/quick_start)
