import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# AgentField with LiteLLM

Use [AgentField](https://agentfield.ai) with any LLM provider through LiteLLM.

AgentField is an open-source control plane for building and orchestrating autonomous AI agents, with SDKs for Python, TypeScript, and Go. AgentField's Python SDK uses LiteLLM internally for multi-provider LLM support.

## Overview

AgentField's Python SDK uses `litellm.acompletion()` under the hood, giving you access to 100+ LLM providers out of the box:

- Use any LiteLLM-supported model (OpenAI, Anthropic, Azure, Bedrock, Ollama, etc.)
- Switch between providers by changing the model string
- All LiteLLM features (caching, fallbacks, routing) work automatically

## Prerequisites

- Python 3.9+
- API keys for your LLM providers
- AgentField control plane (optional, for orchestration features)

## Installation

```bash
pip install agentfield
```

## Quick Start

### Basic Agent with OpenAI

```python
from agentfield import Agent, AgentConfig

config = AgentConfig(
    name="my-agent",
    model="gpt-4o",  # Any LiteLLM-supported model
    instructions="You are a helpful assistant."
)

agent = Agent(config)
response = await agent.run("Hello, world!")
```

### Using Anthropic

```python
config = AgentConfig(
    name="claude-agent",
    model="anthropic/claude-sonnet-4-20250514",  # LiteLLM model format
    instructions="You are a helpful assistant."
)
```

### Using Ollama (Local Models)

```python
config = AgentConfig(
    name="local-agent",
    model="ollama/llama3.1",  # LiteLLM's ollama/ prefix
    instructions="You are a helpful assistant."
)
```

### Using Azure OpenAI

```python
config = AgentConfig(
    name="azure-agent",
    model="azure/gpt-4o",  # LiteLLM's azure/ prefix
    instructions="You are a helpful assistant."
)
```

### Using with LiteLLM Proxy

Point AgentField to a LiteLLM Proxy for centralized model management:

```python
import os

os.environ["OPENAI_API_BASE"] = "http://0.0.0.0:4000"  # LiteLLM Proxy URL
os.environ["OPENAI_API_KEY"] = "sk-1234"  # LiteLLM Proxy key

config = AgentConfig(
    name="proxy-agent",
    model="gpt-4o",  # Virtual model name from proxy config
    instructions="You are a helpful assistant."
)
```

## Multi-Agent Orchestration

AgentField's control plane orchestrates multiple agents, each potentially using different LLM providers:

```python
from agentfield import Agent, AgentConfig, ControlPlane

# Create agents with different providers
researcher = Agent(AgentConfig(
    name="researcher",
    model="anthropic/claude-sonnet-4-20250514",
    instructions="You research topics thoroughly."
))

writer = Agent(AgentConfig(
    name="writer",
    model="gpt-4o",
    instructions="You write clear, concise content."
))

# Register with control plane
cp = ControlPlane(server="http://localhost:8080")
cp.register(researcher)
cp.register(writer)
```

## Links

- [Documentation](https://agentfield.ai/docs)
- [GitHub](https://github.com/Agent-Field/agentfield)
- [Python SDK](https://github.com/Agent-Field/agentfield/tree/main/sdk/python)
