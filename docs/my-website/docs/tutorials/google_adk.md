import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';


# Google ADK with LiteLLM

<Image 
  img={require('../../img/litellm_adk.png')}
  style={{width: '90%', display: 'block', margin: '2rem 0'}}
/>
<p style={{textAlign: 'left', color: '#666'}}>
  Use Google ADK with LiteLLM Python SDK, LiteLLM Proxy
</p>


This tutorial shows you how to create intelligent agents using Agent Development Kit (ADK) with support for multiple Large Language Model (LLM) providers with LiteLLM.



## Overview

ADK (Agent Development Kit) allows you to build intelligent agents powered by LLMs. By integrating with LiteLLM, you can:

- Use multiple LLM providers (OpenAI, Anthropic, Google, etc.)
- Switch easily between models from different providers
- Connect to a LiteLLM proxy for centralized model management

## Prerequisites

- Python environment setup
- API keys for model providers (OpenAI, Anthropic, Google AI Studio)
- Basic understanding of LLMs and agent concepts

## Installation

```bash showLineNumbers title="Install dependencies"
pip install google-adk litellm
```

## 1. Setting Up Environment

First, import the necessary libraries and set up your API keys:

```python showLineNumbers title="Setup environment and API keys"
import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import litellm  # Import for proxy configuration

# Set your API keys
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"  # For Gemini models
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # For OpenAI models
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"  # For Claude models

# Define model constants for cleaner code
MODEL_GEMINI_PRO = "gemini-1.5-pro"
MODEL_GPT_4O = "openai/gpt-4o"
MODEL_CLAUDE_SONNET = "anthropic/claude-3-sonnet-20240229"
```

## 2. Define a Simple Tool

Create a tool that your agent can use:

```python showLineNumbers title="Weather tool implementation"
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.
    
    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").
    
    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    print(f"Tool: get_weather called for city: {city}")
    
    # Mock weather data
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }
    
    city_normalized = city.lower().replace(" ", "")
    
    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}
```

## 3. Helper Function for Agent Interaction

Create a helper function to facilitate agent interaction:

```python showLineNumbers title="Agent interaction helper function"
async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    final_response_text = "Agent did not produce a final response."
    
    # Execute the agent and find the final response
    async for event in runner.run_async(
        user_id=user_id, 
        session_id=session_id, 
        new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            break
            
    print(f"<<< Agent Response: {final_response_text}")
```

## 4. Using Different Model Providers with ADK

### 4.1 Using OpenAI Models

```python showLineNumbers title="OpenAI model implementation"
# Create an agent powered by OpenAI's GPT model
weather_agent_gpt = Agent(
    name="weather_agent_gpt",
    model=LiteLlm(model=MODEL_GPT_4O),  # Use OpenAI's GPT model
    description="Provides weather information using OpenAI's GPT.",
    instruction="You are a helpful weather assistant powered by GPT-4o. "
                "Use the 'get_weather' tool for city weather requests. "
                "Present information clearly.",
    tools=[get_weather],
)

# Set up session and runner
session_service_gpt = InMemorySessionService()
session_gpt = session_service_gpt.create_session(
    app_name="weather_app",
    user_id="user_1",
    session_id="session_gpt"
)

runner_gpt = Runner(
    agent=weather_agent_gpt,
    app_name="weather_app",
    session_service=session_service_gpt
)

# Test the GPT agent
async def test_gpt_agent():
    print("\n--- Testing GPT Agent ---")
    await call_agent_async(
        "What's the weather in London?",
        runner=runner_gpt,
        user_id="user_1",
        session_id="session_gpt"
    )

# Execute the conversation with the GPT agent
await test_gpt_agent()

# Or if running as a standard Python script:
# if __name__ == "__main__":
#     asyncio.run(test_gpt_agent())
```

### 4.2 Using Anthropic Models

```python showLineNumbers title="Anthropic model implementation"
# Create an agent powered by Anthropic's Claude model
weather_agent_claude = Agent(
    name="weather_agent_claude",
    model=LiteLlm(model=MODEL_CLAUDE_SONNET),  # Use Anthropic's Claude model
    description="Provides weather information using Anthropic's Claude.",
    instruction="You are a helpful weather assistant powered by Claude Sonnet. "
                "Use the 'get_weather' tool for city weather requests. "
                "Present information clearly.",
    tools=[get_weather],
)

# Set up session and runner
session_service_claude = InMemorySessionService()
session_claude = session_service_claude.create_session(
    app_name="weather_app",
    user_id="user_1",
    session_id="session_claude"
)

runner_claude = Runner(
    agent=weather_agent_claude,
    app_name="weather_app",
    session_service=session_service_claude
)

# Test the Claude agent
async def test_claude_agent():
    print("\n--- Testing Claude Agent ---")
    await call_agent_async(
        "What's the weather in Tokyo?",
        runner=runner_claude,
        user_id="user_1",
        session_id="session_claude"
    )

# Execute the conversation with the Claude agent
await test_claude_agent()

# Or if running as a standard Python script:
# if __name__ == "__main__":
#     asyncio.run(test_claude_agent())
```

### 4.3 Using Google's Gemini Models

```python showLineNumbers title="Gemini model implementation"
# Create an agent powered by Google's Gemini model
weather_agent_gemini = Agent(
    name="weather_agent_gemini",
    model=MODEL_GEMINI_PRO,  # Use Gemini model directly (no LiteLlm wrapper needed)
    description="Provides weather information using Google's Gemini.",
    instruction="You are a helpful weather assistant powered by Gemini Pro. "
                "Use the 'get_weather' tool for city weather requests. "
                "Present information clearly.",
    tools=[get_weather],
)

# Set up session and runner
session_service_gemini = InMemorySessionService()
session_gemini = session_service_gemini.create_session(
    app_name="weather_app",
    user_id="user_1",
    session_id="session_gemini"
)

runner_gemini = Runner(
    agent=weather_agent_gemini,
    app_name="weather_app",
    session_service=session_service_gemini
)

# Test the Gemini agent
async def test_gemini_agent():
    print("\n--- Testing Gemini Agent ---")
    await call_agent_async(
        "What's the weather in New York?",
        runner=runner_gemini,
        user_id="user_1",
        session_id="session_gemini"
    )

# Execute the conversation with the Gemini agent
await test_gemini_agent()

# Or if running as a standard Python script:
# if __name__ == "__main__":
#     asyncio.run(test_gemini_agent())
```

## 5. Using LiteLLM Proxy with ADK

LiteLLM proxy provides a unified API endpoint for multiple models, simplifying deployment and centralized management.

Required settings for using litellm proxy 

| Variable | Description | 
|----------|-------------|
| `LITELLM_PROXY_API_KEY` | The API key for the LiteLLM proxy |
| `LITELLM_PROXY_API_BASE` | The base URL for the LiteLLM proxy |
| `USE_LITELLM_PROXY` or `litellm.use_litellm_proxy` | When set to True, your request will be sent to litellm proxy. |

```python showLineNumbers title="LiteLLM proxy integration"
# Set your LiteLLM Proxy credentials as environment variables
os.environ["LITELLM_PROXY_API_KEY"] = "your-litellm-proxy-api-key"
os.environ["LITELLM_PROXY_API_BASE"] = "your-litellm-proxy-url"  # e.g., "http://localhost:4000"
# Enable the use_litellm_proxy flag
litellm.use_litellm_proxy = True

# Create a proxy-enabled agent (using environment variables)
weather_agent_proxy_env = Agent(
    name="weather_agent_proxy_env",
    model=LiteLlm(model="gpt-4o"), # this will call the `gpt-4o` model on LiteLLM proxy
    description="Provides weather information using a model from LiteLLM proxy.",
    instruction="You are a helpful weather assistant. "
                "Use the 'get_weather' tool for city weather requests. "
                "Present information clearly.",
    tools=[get_weather],
)

# Set up session and runner
session_service_proxy_env = InMemorySessionService()
session_proxy_env = session_service_proxy_env.create_session(
    app_name="weather_app",
    user_id="user_1",
    session_id="session_proxy_env"
)

runner_proxy_env = Runner(
    agent=weather_agent_proxy_env,
    app_name="weather_app",
    session_service=session_service_proxy_env
)

# Test the proxy-enabled agent (environment variables method)
async def test_proxy_env_agent():
    print("\n--- Testing Proxy-enabled Agent (Environment Variables) ---")
    await call_agent_async(
        "What's the weather in London?",
        runner=runner_proxy_env,
        user_id="user_1",
        session_id="session_proxy_env"
    )

# Execute the conversation
await test_proxy_env_agent()
```
