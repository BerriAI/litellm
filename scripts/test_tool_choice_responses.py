"""Post-fix verification for #23423: tool_choice with responses/ prefix."""
import os
from dotenv import load_dotenv
load_dotenv()

import litellm

# Verify supports_tool_choice resolves correctly
from litellm.utils import supports_tool_choice
print("supports_tool_choice('gpt-5.4'):", supports_tool_choice("gpt-5.4"))
print("supports_tool_choice('openai/responses/gpt-5.4'):", supports_tool_choice("openai/responses/gpt-5.4"))

# Verify tool_choice is in supported params
params = litellm.get_supported_openai_params(model="openai/responses/gpt-5.4", custom_llm_provider="openai")
print("tool_choice in supported params:", "tool_choice" in params)

# Real API call with tool_choice
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

response = litellm.completion(
    model="openai/responses/gpt-4.1-nano",  # cheaper model
    messages=[{"role": "user", "content": "What's the weather in Buenos Aires?"}],
    tools=tools,
    tool_choice="required",
)

print("\nResponse:")
print("  tool_calls:", response.choices[0].message.tool_calls)
print("  finish_reason:", response.choices[0].finish_reason)

has_tool_call = response.choices[0].message.tool_calls is not None
print("\nVERDICT:", "PASS - tool_choice works" if has_tool_call else "FAIL - tool_choice dropped")
