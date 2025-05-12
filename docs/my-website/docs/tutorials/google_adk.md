# Building Multi-Provider Intelligent Agents with ADK and LiteLLM

This tutorial shows you how to create intelligent agents using Agent Development Kit (ADK) with LiteLLM. 

## Introduction

ADK (Agent Development Kit) allows you to build intelligent agents powered by LLMs. By integrating with LiteLLM, you can:

- Use multiple LLM providers (OpenAI, Anthropic, Google, etc.)
- Switch easily between models from different providers
- Connect to a LiteLLM proxy for centralized model management

## Prerequisites

- Python environment setup
- API keys for model providers (OpenAI, Anthropic, Google AI Studio)
- Basic understanding of LLMs and agent concepts

## Installation

```bash
pip install google-adk litellm
```

## 1. Setting Up Environment

First, import the necessary libraries and set up your API keys:

```python
import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

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

```python
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

## 3. Using Different Model Providers with ADK

### 3.1 Using OpenAI Models

```python
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
```

### 3.2 Using Anthropic Models

```python
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
```

### 3.3 Using Google's Gemini Models

```python
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
```

## 4. Interacting with the Agents

Create a helper function to facilitate agent interaction:

```python
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

Interact with each agent:

```python
async def run_conversation():
    # Test with GPT
    print("\n--- Testing GPT Agent ---")
    await call_agent_async(
        "What's the weather in London?",
        runner=runner_gpt,
        user_id="user_1",
        session_id="session_gpt"
    )
    
    # Test with Claude
    print("\n--- Testing Claude Agent ---")
    await call_agent_async(
        "What's the weather in Tokyo?",
        runner=runner_claude,
        user_id="user_1",
        session_id="session_claude"
    )
    
    # Test with Gemini
    print("\n--- Testing Gemini Agent ---")
    await call_agent_async(
        "What's the weather in New York?",
        runner=runner_gemini,
        user_id="user_1",
        session_id="session_gemini"
    )

# Execute the conversation
await run_conversation()

# Or if running as a standard Python script:
# if __name__ == "__main__":
#     asyncio.run(run_conversation())
```

## 5. Using LiteLLM Proxy with ADK

LiteLLM proxy provides a unified API endpoint for multiple models, simplifying deployment and centralized management.

### 5.1 Setting Up LiteLLM Proxy Environment

```python
# Set your LiteLLM Proxy credentials
os.environ["LITELLM_PROXY_API_KEY"] = "your-litellm-proxy-api-key"
os.environ["LITELLM_PROXY_API_BASE"] = "your-litellm-proxy-url"  # e.g., "http://localhost:4000"

# Alternatively, set them dynamically without environment variables:
LITELLM_PROXY_API_KEY = "your-litellm-proxy-api-key"
LITELLM_PROXY_API_BASE = "your-litellm-proxy-url"
```

### 5.2 Creating an Agent Using LiteLLM Proxy

```python
# Proxy-enabled agent with dynamic credentials
weather_agent_proxy = Agent(
    name="weather_agent_proxy",
    model=LiteLlm(
        model="litellm_proxy/your-model-name",  # The model name registered in your proxy
        api_key=LITELLM_PROXY_API_KEY,          # Dynamic API key
        api_base=LITELLM_PROXY_API_BASE         # Dynamic API base URL
    ),
    description="Provides weather information using a model from LiteLLM proxy.",
    instruction="You are a helpful weather assistant. "
                "Use the 'get_weather' tool for city weather requests. "
                "Present information clearly.",
    tools=[get_weather],
)

# Setup session and runner
session_service_proxy = InMemorySessionService()
session_proxy = session_service_proxy.create_session(
    app_name="weather_app",
    user_id="user_1",
    session_id="session_proxy"
)

runner_proxy = Runner(
    agent=weather_agent_proxy,
    app_name="weather_app",
    session_service=session_service_proxy
)

# Test the proxy-enabled agent
async def test_proxy_agent():
    print("\n--- Testing Proxy-enabled Agent ---")
    await call_agent_async(
        "What's the weather in London?",
        runner=runner_proxy,
        user_id="user_1",
        session_id="session_proxy"
    )

await test_proxy_agent()
```

