---
slug: minimax_m2_5
title: "Day 0 Support: MiniMax-M2.5"
date: 2026-02-12T10:00:00
authors:
  - name: Sameer Kankute
    title: SWE @ LiteLLM (LLM Translation)
    url: https://www.linkedin.com/in/sameer-kankute/
    image_url: https://pbs.twimg.com/profile_images/2001352686994907136/ONgNuSk5_400x400.jpg
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
description: "Day 0 support for MiniMax-M2.5 on LiteLLM"
tags: [minimax, M2.5, llm]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

LiteLLM now supports MiniMax-M2.5 on Day 0. Use it across OpenAI-compatible and Anthropic-compatible APIs through the LiteLLM AI Gateway.

## Supported Models

LiteLLM supports the following MiniMax models:

| Model | Description | Input Cost | Output Cost | Context Window |
|-------|-------------|------------|-------------|----------------|
| **MiniMax-M2.5** | Advanced reasoning, Agentic capabilities | $0.3/M tokens | $1.2/M tokens | 1M tokens |
| **MiniMax-M2.5-lightning** | Faster and More Agile (~100 tps) | $0.3/M tokens | $2.4/M tokens | 1M tokens |

## Features Supported

- **Prompt Caching**: Reduce costs with cached prompts ($0.03/M tokens for cache read, $0.375/M tokens for cache write)
- **Function Calling**: Built-in tool calling support
- **Reasoning**: Advanced reasoning capabilities with thinking support
- **System Messages**: Full system message support
- **Cost Tracking**: Automatic cost calculation for all requests

## Docker Image

```bash
docker pull litellm/litellm:v1.81.3-stable
```

## Usage - OpenAI Compatible API (/v1/chat/completions)

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: minimax-m2-5
    litellm_params:
      model: minimax/MiniMax-M2.5
      api_key: os.environ/MINIMAX_API_KEY
      api_base: https://api.minimax.io/v1
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e MINIMAX_API_KEY=$MINIMAX_API_KEY \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:v1.81.3-stable \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "minimax-m2-5",
  "messages": [
    {
      "role": "user",
      "content": "what llm are you"
    }
  ]
}'
```

</TabItem>
</Tabs>

### With Reasoning Split

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "minimax-m2-5",
  "messages": [
    {
      "role": "user",
      "content": "Solve: 2+2=?"
    }
  ],
  "extra_body": {
    "reasoning_split": true
  }
}'
```

## Usage - Anthropic Compatible API (/v1/messages)

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: minimax-m2-5
    litellm_params:
      model: minimax/MiniMax-M2.5
      api_key: os.environ/MINIMAX_API_KEY
      api_base: https://api.minimax.io/anthropic/v1/messages
```

**2. Start the proxy**

```bash
docker run -d \
  -p 4000:4000 \
  -e MINIMAX_API_KEY=$MINIMAX_API_KEY \
  -v $(pwd)/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:v1.81.3-stable \
  --config /app/config.yaml
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "minimax-m2-5",
  "max_tokens": 1000,
  "messages": [
    {
      "role": "user",
      "content": "what llm are you"
    }
  ]
}'
```

</TabItem>
</Tabs>

### With Thinking

```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data '{
  "model": "minimax-m2-5",
  "max_tokens": 1000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 1000
  },
  "messages": [
    {
      "role": "user",
      "content": "Solve: 2+2=?"
    }
  ]
}'
```

## Usage - LiteLLM SDK

### OpenAI-compatible API

```python
import litellm

response = litellm.completion(
    model="minimax/MiniMax-M2.5",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    api_key="your-minimax-api-key",
    api_base="https://api.minimax.io/v1"
)

print(response.choices[0].message.content)
```

### Anthropic-compatible API

```python
import litellm

response = litellm.anthropic.messages.acreate(
    model="minimax/MiniMax-M2.5",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    api_key="your-minimax-api-key",
    api_base="https://api.minimax.io/anthropic/v1/messages",
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### With Thinking

```python
response = litellm.anthropic.messages.acreate(
    model="minimax/MiniMax-M2.5",
    messages=[{"role": "user", "content": "Solve: 2+2=?"}],
    thinking={"type": "enabled", "budget_tokens": 1000},
    api_key="your-minimax-api-key"
)

# Access thinking content
for block in response.choices[0].message.content:
    if hasattr(block, 'type') and block.type == 'thinking':
        print(f"Thinking: {block.thinking}")
```

### With Reasoning Split (OpenAI API)

```python
response = litellm.completion(
    model="minimax/MiniMax-M2.5",
    messages=[
        {"role": "user", "content": "Solve: 2+2=?"}
    ],
    extra_body={"reasoning_split": True},
    api_key="your-minimax-api-key",
    api_base="https://api.minimax.io/v1"
)

# Access thinking and response
if hasattr(response.choices[0].message, 'reasoning_details'):
    print(f"Thinking: {response.choices[0].message.reasoning_details}")
print(f"Response: {response.choices[0].message.content}")
```

## Cost Tracking

LiteLLM automatically tracks costs for MiniMax-M2.5 requests. The pricing is:

- **Input**: $0.3 per 1M tokens
- **Output**: $1.2 per 1M tokens
- **Cache Read**: $0.03 per 1M tokens
- **Cache Write**: $0.375 per 1M tokens

### Accessing Cost Information

```python
response = litellm.completion(
    model="minimax/MiniMax-M2.5",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="your-minimax-api-key"
)

# Access cost information
print(f"Cost: ${response._hidden_params.get('response_cost', 0)}")
```

## Streaming Support

### OpenAI API

```python
response = litellm.completion(
    model="minimax/MiniMax-M2.5",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    api_key="your-minimax-api-key",
    api_base="https://api.minimax.io/v1"
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Streaming with Reasoning Split

```python
stream = litellm.completion(
    model="minimax/MiniMax-M2.5",
    messages=[
        {"role": "user", "content": "Tell me a story"},
    ],
    extra_body={"reasoning_split": True},
    stream=True,
    api_key="your-minimax-api-key",
    api_base="https://api.minimax.io/v1"
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

## Using with Native SDKs

### Anthropic SDK via LiteLLM Proxy

```python
import os
os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:4000"
os.environ["ANTHROPIC_API_KEY"] = "sk-1234"  # Your LiteLLM proxy key

import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="minimax-m2-5",
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

### OpenAI SDK via LiteLLM Proxy

```python
import os
os.environ["OPENAI_BASE_URL"] = "http://localhost:4000"
os.environ["OPENAI_API_KEY"] = "sk-1234"  # Your LiteLLM proxy key

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="minimax-m2-5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, how are you?"},
    ],
    extra_body={"reasoning_split": True},
)

# Access thinking and response
if hasattr(response.choices[0].message, 'reasoning_details'):
    print(f"Thinking:\n{response.choices[0].message.reasoning_details[0]['text']}\n")
print(f"Text:\n{response.choices[0].message.content}\n")
```
