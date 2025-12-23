import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MiniMax - v1/messages

## Overview

Litellm provides anthropic specs compatible support for minmax

## Supported Models

MiniMax offers three models through their Anthropic-compatible API:

| Model | Description | Input Cost | Output Cost | Prompt Caching Read | Prompt Caching Write |
|-------|-------------|------------|-------------|---------------------|----------------------|
| **MiniMax-M2.1** | Powerful Multi-Language Programming with Enhanced Programming Experience (~60 tps) | $0.3/M tokens | $1.2/M tokens | $0.03/M tokens | $0.375/M tokens |
| **MiniMax-M2.1-lightning** | Faster and More Agile (~100 tps) | $0.3/M tokens | $2.4/M tokens | $0.03/M tokens | $0.375/M tokens |
| **MiniMax-M2** | Agentic capabilities, Advanced reasoning | $0.3/M tokens | $1.2/M tokens | $0.03/M tokens | $0.375/M tokens |


## Usage Examples

### Basic Chat Completion

```python
import litellm

response = litellm.anthropic.messages.acreate(
    model="minimax/MiniMax-M2.1",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    api_key="your-minimax-api-key",
    api_base="https://api.minimax.io/anthropic/v1/messages",
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### Using Environment Variables

```bash
export MINIMAX_API_KEY="your-minimax-api-key"
export MINIMAX_API_BASE="https://api.minimax.io/anthropic/v1/messages"
```

```python
import litellm

response = litellm.anthropic.messages.acreate(
    model="minimax/MiniMax-M2.1",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=1000
)
```

### With Thinking (M2.1 Feature)

```python
response = litellm.anthropic.messages.acreate(
    model="minimax/MiniMax-M2.1",
    messages=[{"role": "user", "content": "Solve: 2+2=?"}],
    thinking={"type": "enabled", "budget_tokens": 1000},
    api_key="your-minimax-api-key"
)

# Access thinking content
for block in response.choices[0].message.content:
    if hasattr(block, 'type') and block.type == 'thinking':
        print(f"Thinking: {block.thinking}")
```

### With Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = litellm.anthropic.messages.acreate(
    model="minimax/MiniMax-M2.1",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools,
    api_key="your-minimax-api-key",
    max_tokens=1000
)
```



## Usage with LiteLLM Proxy 

You can use MiniMax models with the Anthropic SDK by routing through LiteLLM Proxy:

| Step | Description |
|------|-------------|
| **1. Start LiteLLM Proxy** | Configure proxy with MiniMax models in `config.yaml` |
| **2. Set Environment Variables** | Point Anthropic SDK to proxy endpoint |
| **3. Use Anthropic SDK** | Call MiniMax models using native Anthropic SDK |

### Step 1: Configure LiteLLM Proxy

Create a `config.yaml`:

```yaml
model_list:
  - model_name: minimax/MiniMax-M2.1
    litellm_params:
      model: minimax/MiniMax-M2.1
      api_key: os.environ/MINIMAX_API_KEY
      api_base: https://api.minimax.io/anthropic/v1/messages
```

Start the proxy:

```bash
litellm --config config.yaml
```

### Step 2: Use with Anthropic SDK

```python
import os
os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:4000"
os.environ["ANTHROPIC_API_KEY"] = "sk-1234"  # Your LiteLLM proxy key

import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="minimax/MiniMax-M2.1",
    max_tokens=1000,
    system="You are a helpful assistant.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hi, how are you?"
                }
            ]
        }
    ]
)

for block in message.content:
    if block.type == "thinking":
        print(f"Thinking:\n{block.thinking}\n")
    elif block.type == "text":
        print(f"Text:\n{block.text}\n")
```
## Cost Calculation

Cost calculation works automatically using the pricing information in `model_prices_and_context_window.json`.

Example:
```python
response = litellm.completion(
    model="minimax/MiniMax-M2.1",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="your-minimax-api-key"
)

# Access cost information
print(f"Cost: ${response._hidden_params.get('response_cost', 0)}")
```


