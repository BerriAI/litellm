import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MiniMax  

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

# MiniMax - v1/chat/completions

## Usage with LiteLLM SDK

You can use MiniMax's OpenAI-compatible API directly with LiteLLM:

### Basic Chat Completion

```python
import litellm

response = litellm.completion(
    model="minimax/MiniMax-M2.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    api_key="your-minimax-api-key",
    api_base="https://api.minimax.io/v1"
)

print(response.choices[0].message.content)
```

### Using Environment Variables

```bash
export MINIMAX_API_KEY="your-minimax-api-key"
export MINIMAX_API_BASE="https://api.minimax.io/v1"
```

```python
import litellm

response = litellm.completion(
    model="minimax/MiniMax-M2.1",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### With Reasoning Split

```python
response = litellm.completion(
    model="minimax/MiniMax-M2.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Solve: 2+2=?"}
    ],
    extra_body={"reasoning_split": True},
    api_key="your-minimax-api-key",
    api_base="https://api.minimax.io/v1"
)

# Access reasoning details if available
if hasattr(response.choices[0].message, 'reasoning_details'):
    print(f"Thinking: {response.choices[0].message.reasoning_details}")
print(f"Response: {response.choices[0].message.content}")
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

response = litellm.completion(
    model="minimax/MiniMax-M2.1",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools,
    api_key="your-minimax-api-key",
    api_base="https://api.minimax.io/v1"
)
```

### Streaming

```python
response = litellm.completion(
    model="minimax/MiniMax-M2.1",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    api_key="your-minimax-api-key",
    api_base="https://api.minimax.io/v1"
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```


## Usage with OpenAI SDK via LiteLLM Proxy

You can also use MiniMax models with the OpenAI SDK by routing through LiteLLM Proxy:

| Step | Description |
|------|-------------|
| **1. Start LiteLLM Proxy** | Configure proxy with MiniMax models in `config.yaml` |
| **2. Set Environment Variables** | Point OpenAI SDK to proxy endpoint |
| **3. Use OpenAI SDK** | Call MiniMax models using native OpenAI SDK |

### Step 1: Configure LiteLLM Proxy

Create a `config.yaml`:

```yaml
model_list:
  - model_name: minimax/MiniMax-M2.1
    litellm_params:
      model: minimax/MiniMax-M2.1
      api_key: os.environ/MINIMAX_API_KEY
      api_base: https://api.minimax.io/v1
```

Start the proxy:

```bash
litellm --config config.yaml
```

### Step 2: Use with OpenAI SDK

```python
import os
os.environ["OPENAI_BASE_URL"] = "http://localhost:4000"
os.environ["OPENAI_API_KEY"] = "sk-1234"  # Your LiteLLM proxy key

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="minimax/MiniMax-M2.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, how are you?"},
    ],
    # Set reasoning_split=True to separate thinking content
    extra_body={"reasoning_split": True},
)

# Access thinking and response
if hasattr(response.choices[0].message, 'reasoning_details'):
    print(f"Thinking:\n{response.choices[0].message.reasoning_details[0]['text']}\n")
print(f"Text:\n{response.choices[0].message.content}\n")
```

### Streaming with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="minimax/MiniMax-M2.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    extra_body={"reasoning_split": True},
    stream=True,
)

reasoning_buffer = ""
text_buffer = ""

for chunk in stream:
    if hasattr(chunk.choices[0].delta, "reasoning_details") and chunk.choices[0].delta.reasoning_details:
        for detail in chunk.choices[0].delta.reasoning_details:
            if "text" in detail:
                reasoning_text = detail["text"]
                new_reasoning = reasoning_text[len(reasoning_buffer):]
                if new_reasoning:
                    print(new_reasoning, end="", flush=True)
                    reasoning_buffer = reasoning_text

    if chunk.choices[0].delta.content:
        content_text = chunk.choices[0].delta.content
        new_text = content_text[len(text_buffer):] if text_buffer else content_text
        if new_text:
            print(new_text, end="", flush=True)
            text_buffer = content_text
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


