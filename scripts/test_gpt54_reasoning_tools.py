"""
Repro script: verify that gpt-5.4 drops reasoning_effort when tools are present.
Expected: the call succeeds (reasoning_effort is silently dropped).
If the bug were still present, OpenAI would return an error like:
  "reasoning_effort is not supported with function calling"
"""

import os
from dotenv import load_dotenv

load_dotenv()

import litellm

litellm.set_verbose = True

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]

print("=== Test: gpt-5.4 + reasoning_effort='medium' + tools ===")
try:
    response = litellm.completion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "What's the weather in Buenos Aires?"}],
        reasoning_effort="medium",
        tools=tools,
        drop_params=True,
    )
    print(f"SUCCESS - model: {response.model}")
    print(f"Choice: {response.choices[0].message}")
    if response.choices[0].message.tool_calls:
        print(f"Tool calls: {response.choices[0].message.tool_calls}")
    print("\nreasoning_effort was correctly dropped (no error from OpenAI)")
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== Test: gpt-5.4 + reasoning_effort='high' + tools ===")
try:
    response = litellm.completion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "What's 2+2?"}],
        reasoning_effort="high",
        tools=tools,
        drop_params=True,
    )
    print(f"SUCCESS - model: {response.model}")
    print(f"reasoning_effort was correctly dropped (no error from OpenAI)")
except Exception as e:
    print(f"FAILED: {e}")

print("\n=== Test: gpt-5.4 + reasoning_effort='none' + tools (should KEEP reasoning_effort) ===")
try:
    response = litellm.completion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Say hello"}],
        reasoning_effort="none",
        tools=tools,
        drop_params=True,
    )
    print(f"SUCCESS - model: {response.model}")
    print(f"reasoning_effort='none' correctly kept (OpenAI allows this)")
except Exception as e:
    print(f"FAILED: {e}")
