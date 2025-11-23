import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Using OpenAI Agents with LiteLLM

OpenAI Agents is a powerful framework for building AI agents that can use tools and perform complex tasks. When combined with LiteLLM, you can extend agent capabilities to work with 100+ LLM providers while maintaining OpenAI's agent framework.

:::tip
Check out this tutorial on integrating LiteLLM with OpenAI Agents:

📹 **[OpenAI Agents + LiteLLM Tutorial](https://screen.studio/share/THDYAjd4)**
:::

## Pre-Requisites

```shell
pip install openai-agents[litellm] litellm
```

## Quick Start

### Basic OpenAI Agent with LiteLLM

<Tabs>
<TabItem value="litellm-proxy" label="OpenAI">

```python
from __future__ import annotations

import asyncio
import os

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

@function_tool
def get_weather(city: str):
    """Get the current weather for a city."""
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."

async def main(model: str):
    """Run the agent with the specified model."""
    
    llm = LitellmModel(
        model=model,
        api_key="sk-1234",
        base_url="http://localhost:4000",  # LiteLLM proxy URL
    )

    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=llm,
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    # Use a model configured in your LiteLLM proxy
    asyncio.run(main("gpt-5-codex"))
```

</TabItem>

<TabItem value="anthropic" label="Anthropic">

```python
import asyncio
import os

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 22°C."

async def main():
    """Run agent with Anthropic Claude."""
    
    llm = LitellmModel(
        model="claude-3-sonnet-20240229",
        api_key="sk-1234",
        base_url="http://localhost:4000",  # LiteLLM proxy URL
    )

    agent = Agent(
        name="Weather Assistant",
        instructions="You are a helpful weather assistant. Respond in French.",
        model=llm,
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Paris?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

</TabItem>

<TabItem value="gemini" label="Gemini">

```python
import asyncio
import os

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is clear."

async def main():
    """Run agent with Groq."""
    
    llm = LitellmModel(
        model="mixtral-8x7b-32768",
        api_key=os.getenv("GROQ_API_KEY"),
    )

    agent = Agent(
        name="Weather Assistant",
        instructions="You are a helpful weather assistant.",
        model=llm,
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

</TabItem>
</Tabs>

## Agent Patterns with OpenAI Agents SDK

### Multi-Tool Agent

```python
import asyncio
import os

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."

@function_tool
def get_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

async def main():
    """Run agent with multiple tools."""
    
    llm = LitellmModel(
        model="gpt-4o",
        api_key=os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        base_url="http://localhost:4000",
    )

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant. Use tools to answer questions accurately.",
        model=llm,
        tools=[get_weather, get_time],  # Multiple tools
    )

    result = await Runner.run(
        agent,
        "What's the weather in Tokyo? Also, what time is it?"
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

### Agent with Custom Parameters

```python
import asyncio
import os
from typing import Annotated

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

@function_tool
def calculate(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
    """
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
    }
    return ops.get(operation, "Unknown operation")(a, b)

async def main():
    """Run agent with parameter annotations."""
    
    llm = LitellmModel(
        model="claude-3-sonnet-20240229",
    )

    agent = Agent(
        name="Calculator",
        instructions="You are a helpful calculator assistant.",
        model=llm,
        tools=[calculate],
    )

    result = await Runner.run(agent, "What is 42 divided by 7?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

## Agent Patterns

### Stateful Agent with Memory

```python
import asyncio
import os
from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

# Shared memory store
memory_store = {}

@function_tool
def remember(key: str, value: str) -> str:
    """Store information in memory."""
    memory_store[key] = value
    return f"Remembered: {key} = {value}"

@function_tool
def recall(key: str) -> str:
    """Retrieve information from memory."""
    if key in memory_store:
        return f"You told me: {memory_store[key]}"
    return f"I don't remember anything about {key}"

async def main():
    """Run agent with stateful memory."""
    
    llm = LitellmModel(
        model="gpt-4o",
        api_key=os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        base_url="http://localhost:4000",
    )

    agent = Agent(
        name="Memory Assistant",
        instructions="You are a helpful assistant with memory. Store information when asked and recall it when needed.",
        model=llm,
        tools=[remember, recall],
    )

    # First interaction: store information
    result1 = await Runner.run(agent, "Remember that my favorite color is blue")
    print(result1.final_output)
    
    # Second interaction: retrieve information
    result2 = await Runner.run(agent, "What is my favorite color?")
    print(result2.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```



## Resources

- [OpenAI Agents Documentation](https://platform.openai.com/docs/guides/agents)
- [LiteLLM Proxy Server](../proxy_server.md)
- [LiteLLM Router & Fallbacks](../router_architecture.md)
- [LiteLLM with LangGraph](./langgraph.md)
