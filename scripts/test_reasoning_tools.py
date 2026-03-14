"""Test gpt-5.4 with reasoning_effort + tools to see OpenAI's response."""
import os
from dotenv import load_dotenv
load_dotenv()

import litellm

try:
    response = litellm.completion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "What's the weather in SF?"}],
        tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        reasoning_effort="high",
    )
    print("SUCCESS:")
    print(response)
except Exception as e:
    print(f"ERROR ({type(e).__name__}):")
    print(e)
