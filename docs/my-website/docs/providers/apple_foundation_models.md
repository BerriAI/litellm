import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Apple Foundation Models

LiteLLM supports Apple's on-device Foundation Models available on macOS 26+.

## Pre-requisites

- macOS 26.0+ (Sequoia) with Apple Intelligence enabled
- Install: `pip install apple-foundation-models`

## Quick Start

```python
from litellm import completion

response = completion(
    model="apple_foundation_models/system",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    max_tokens=100
)
print(response)
```

## Streaming

Enable streaming to receive responses token-by-token:

```python
from litellm import completion

response = completion(
    model="apple_foundation_models/system",
    messages=[{"role": "user", "content": "Write a short poem about AI"}],
    stream=True,
    max_tokens=200
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

## Async

Use `acompletion` for async operations:

<Tabs>
<TabItem value="non-streaming" label="Async (Non-Streaming)">

```python
import asyncio
from litellm import acompletion

async def main():
    response = await acompletion(
        model="apple_foundation_models/system",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=100
    )
    print(response)

asyncio.run(main())
```

</TabItem>
<TabItem value="streaming" label="Async + Streaming">

```python
import asyncio
from litellm import acompletion

async def main():
    response = await acompletion(
        model="apple_foundation_models/system",
        messages=[{"role": "user", "content": "Write a short poem about AI"}],
        stream=True,
        max_tokens=200
    )

    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)

asyncio.run(main())
```

</TabItem>
</Tabs>

## Tool Calling

apple-foundation-models can introspect Python functions directly. Just pass your functions with proper docstrings and type hints:

```python
from litellm import completion

# Define your tool functions with docstrings and type hints
def get_weather(location: str, units: str = "celsius") -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}: 22°{units[0].upper()}, sunny"

def calculate(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

response = completion(
    model="apple_foundation_models/system",
    messages=[{"role": "user", "content": "What's the weather in Paris and what's 5 plus 7?"}],
    tool_functions=[get_weather, calculate],  # Just pass the functions!
    max_tokens=200
)

# Check which tools were called
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Tool: {tool_call.function.name}({tool_call.function.arguments})")

print(response.choices[0].message.content)
```

<Tabs>
<TabItem value="openai-format" label="OpenAI Format (Optional)">

For compatibility with other providers, you can also use OpenAI-style schemas:

```python
from litellm import completion

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}: 22°C, sunny"

# OpenAI-style tool schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

# Map function names to implementations
tool_functions = {
    "get_weather": get_weather
}

response = completion(
    model="apple_foundation_models/system",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,  # OpenAI schemas
    tool_functions=tool_functions,  # Implementations
    max_tokens=200
)
```

</TabItem>
</Tabs>

## Structured Output

Apple Foundation Models supports structured output with both Pydantic models and JSON schemas.

<Tabs>
<TabItem value="pydantic" label="Pydantic Model">

```python
from litellm import completion
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

response = completion(
    model="apple_foundation_models/system",
    messages=[
        {
            "role": "user",
            "content": "Extract person info: Alice is 30 and lives in Paris."
        }
    ],
    response_format=Person,  # Pass Pydantic model directly
    max_tokens=150
)

# Response is automatically formatted as JSON
import json
data = json.loads(response.choices[0].message.content)
print(data)  # {'name': 'Alice', 'age': 30, 'city': 'Paris'}
```

</TabItem>
<TabItem value="json-schema" label="JSON Schema">

```python
from litellm import completion
import json

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"}
    },
    "required": ["name", "age"]
}

response = completion(
    model="apple_foundation_models/system",
    messages=[
        {
            "role": "user",
            "content": "Extract person info: Alice is 30 and lives in Paris. Return as JSON."
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {"schema": schema}
    },
    max_tokens=150
)

# Parse the JSON response
data = json.loads(response.choices[0].message.content)
print(data)  # {'name': 'Alice', 'age': 30, 'city': 'Paris'}
```

</TabItem>
</Tabs>

## Supported Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `temperature` | float | Controls randomness (0.0-2.0) | 1.0 |
| `max_tokens` | int | Maximum tokens to generate | 1024 |
| `stream` | bool | Enable streaming | False |
| `tools` | list | Tool/function definitions (OpenAI format) | None |
| `tool_functions` | list or dict | List of callables or dict mapping names to callables | None |
| `response_format` | dict | JSON schema for structured output | None |

## Troubleshooting

**apple-foundation-models not available:**

- Verify macOS 26.0+: `sw_vers -productVersion`
- Check System Settings that Apple Intelligence  is enabled

## Links

- [apple-foundation-models-py Package](https://github.com/btucker/apple-foundation-models-py)
- [Apple FoundationModels Framework](https://developer.apple.com/documentation/FoundationModels)
- [Apple Intelligence Documentation](https://developer.apple.com/apple-intelligence/)
