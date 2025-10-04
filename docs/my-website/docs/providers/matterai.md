import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MatterAI

https://docs.matterai.so

MatterAI offers SuperIntelligent large language models for general purpose, coding and research. Its OpenAI-compatible API makes integration straightforward, enabling developers to build efficient and scalable AI applications.

| Property                  | Details                                                                     |
| ------------------------- | --------------------------------------------------------------------------- |
| Description               | MatterAI offers powerful language models like `axon-base` and `axon-code`.  |
| Provider Route on LiteLLM | `matterai/` (add this prefix to the model name - e.g. `matterai/axon-base`) |
| Provider Doc              | [MatterAI ↗](https://docs.matterai.so)                                      |
| API Endpoint for Provider | https://api.matterai.so/v1                                                  |
| Supported Endpoints       | `/chat/completions`, `/completions`                                         |

## Supported OpenAI Parameters

MatterAI is fully OpenAI-compatible and supports the following parameters:

```
"stream",
"stop",
"temperature",
"top_p",
"max_tokens",
"presence_penalty",
"frequency_penalty",
"logit_bias",
"user",
"response_format",
"seed",
"tools",
"tool_choice",
"parallel_tool_calls",
"extra_headers"
```

## API Key Setup

To use MatterAI, set your API key as an environment variable:

```python
import os

os.environ["MATTERAI_API_KEY"] = "your-api-key"
```

## Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ['MATTERAI_API_KEY'] = "your-api-key"

response = completion(
    model="matterai/axon-base",
    messages=[
       {"role": "user", "content": "Hello from LiteLLM!"}
   ],
)
print(response)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

```yaml
model_list:
  - model_name: matterai-axon-base
    litellm_params:
      model: matterai/axon-base
      api_key: os.environ/MATTERAI_API_KEY
```

</TabItem>
</Tabs>

## Streaming

```python
from litellm import completion
import os

os.environ['MATTERAI_API_KEY'] = "your-api-key"

response = completion(
    model="matterai/axon-code",
    messages=[
       {"role": "user", "content": "Write a short story about a robot learning to code."}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Advanced Usage

### Custom Parameters

```python
from litellm import completion

response = completion(
    model="matterai/axon-base",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
    stop=["Human:", "AI:"]
)
```

### Function Calling

MatterAI supports OpenAI-compatible function calling:

```python
from litellm import completion

functions = [
    {
        "name": "get_weather",
        "description": "Get current weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state"
                }
            },
            "required": ["location"]
        }
    }
]

response = completion(
    model="matterai/axon-base",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=[{"type": "function", "function": f} for f in functions],
    tool_choice="auto"
)
```

### Async Usage

```python
import asyncio
from litellm import acompletion

async def async_call():
    response = await acompletion(
        model="matterai/axon-base",
        messages=[{"role": "user", "content": "Hello async world!"}]
    )
    return response

# Run async function
response = asyncio.run(async_call())
print(response)
```

## Available Models

MatterAI offers models like `axon-base` and `axon-code`.

Common model formats:

- `matterai/axon-base`
- `matterai/axon-code`

## Benefits

- **Powerful Models**: Access to advanced language models optimized for various tasks
- **OpenAI Compatibility**: Seamless integration with existing OpenAI-compatible tools and workflows
- **Scalable**: Built for efficient, high-throughput applications
- **Developer-Friendly**: Simple API with comprehensive documentation

## Error Handling

MatterAI returns standard OpenAI-compatible error responses:

```python
from litellm import completion
from litellm.exceptions import AuthenticationError, RateLimitError

try:
    response = completion(
        model="matterai/axon-base",
        messages=[{"role": "user", "content": "Hello"}]
    )
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
```

## Support

- Documentation: https://api.matterai.so
- Contact: support@matterai.so
