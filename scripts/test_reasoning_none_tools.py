"""
Repro: gpt-5.4 + reasoning_effort='none' + tools
Current behavior: reasoning_effort='none' is NOT dropped, but OpenAI rejects it.
"""

import os
from dotenv import load_dotenv

load_dotenv()

import litellm

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

print("=== gpt-5.4 + reasoning_effort='none' + tools ===")
try:
    response = litellm.completion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "What's the weather in Buenos Aires?"}],
        reasoning_effort="none",
        tools=tools,
    )
    print(f"SUCCESS - model: {response.model}")
    print(f"Choice: {response.choices[0].message}")
except Exception as e:
    print(f"FAILED: {e}")
