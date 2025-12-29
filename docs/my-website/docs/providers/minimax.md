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

# MiniMax - Text-to-Speech

## Quick Start

## **LiteLLM Python SDK Usage**

### Basic Usage

```python
from pathlib import Path
from litellm import speech
import os 

os.environ["MINIMAX_API_KEY"] = "your-api-key"

speech_file_path = Path(__file__).parent / "speech.mp3"
response = speech(
    model="minimax/speech-2.6-hd",
    voice="alloy",
    input="The quick brown fox jumped over the lazy dogs",
)
response.stream_to_file(speech_file_path)
```

### Async Usage

```python
from litellm import aspeech
from pathlib import Path
import os, asyncio

os.environ["MINIMAX_API_KEY"] = "your-api-key"

async def test_async_speech(): 
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = await aspeech(
        model="minimax/speech-2.6-hd",
        voice="alloy",
        input="The quick brown fox jumped over the lazy dogs",
    )
    response.stream_to_file(speech_file_path)

asyncio.run(test_async_speech())
```

### Voice Selection

MiniMax supports many voices. LiteLLM provides OpenAI-compatible voice names that map to MiniMax voices:

```python
from litellm import speech

# OpenAI-compatible voice names
voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

for voice in voices:
    response = speech(
        model="minimax/speech-2.6-hd",
        voice=voice,
        input=f"This is the {voice} voice",
    )
    response.stream_to_file(f"speech_{voice}.mp3")
```

You can also use MiniMax-native voice IDs directly:

```python
response = speech(
    model="minimax/speech-2.6-hd",
    voice="male-qn-qingse",  # MiniMax native voice ID
    input="Using native MiniMax voice ID",
)
```

### Custom Parameters

MiniMax TTS supports additional parameters for fine-tuning audio output:

```python
from litellm import speech

response = speech(
    model="minimax/speech-2.6-hd",
    voice="alloy",
    input="Custom audio parameters",
    speed=1.5,  # Speed: 0.5 to 2.0
    response_format="mp3",  # Format: mp3, pcm, wav, flac
    extra_body={
        "vol": 1.2,  # Volume: 0.1 to 10
        "pitch": 2,  # Pitch adjustment: -12 to 12
        "sample_rate": 32000,  # 16000, 24000, or 32000
        "bitrate": 128000,  # For MP3: 64000, 128000, 192000, 256000
        "channel": 1,  # 1 for mono, 2 for stereo
    }
)
response.stream_to_file("custom_speech.mp3")
```

### Response Formats

```python
from litellm import speech

# MP3 format (default)
response = speech(
    model="minimax/speech-2.6-hd",
    voice="alloy",
    input="MP3 format audio",
    response_format="mp3",
)

# PCM format
response = speech(
    model="minimax/speech-2.6-hd",
    voice="alloy",
    input="PCM format audio",
    response_format="pcm",
)

# WAV format
response = speech(
    model="minimax/speech-2.6-hd",
    voice="alloy",
    input="WAV format audio",
    response_format="wav",
)

# FLAC format
response = speech(
    model="minimax/speech-2.6-hd",
    voice="alloy",
    input="FLAC format audio",
    response_format="flac",
)
```

## **LiteLLM Proxy Usage**

LiteLLM provides an OpenAI-compatible `/audio/speech` endpoint for MiniMax TTS.

### Setup

Add MiniMax to your proxy configuration:

```yaml
model_list:
  - model_name: tts
    litellm_params:
      model: minimax/speech-2.6-hd
      api_key: os.environ/MINIMAX_API_KEY
  
  - model_name: tts-turbo
    litellm_params:
      model: minimax/speech-2.6-turbo
      api_key: os.environ/MINIMAX_API_KEY
```

Start the proxy:

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### Making Requests

```bash
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "alloy"
  }' \
  --output speech.mp3
```

With custom parameters:

```bash
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts",
    "input": "Custom parameters example.",
    "voice": "nova",
    "speed": 1.5,
    "response_format": "mp3",
    "extra_body": {
      "vol": 1.2,
      "pitch": 1,
      "sample_rate": 32000
    }
  }' \
  --output custom_speech.mp3
```

## Voice Mappings

LiteLLM maps OpenAI-compatible voice names to MiniMax voice IDs:

| OpenAI Voice | MiniMax Voice ID | Description |
|--------------|------------------|-------------|
| alloy | male-qn-qingse | Male voice |
| echo | male-qn-jingying | Male voice |
| fable | female-shaonv | Female voice |
| onyx | male-qn-badao | Male voice |
| nova | female-yujie | Female voice |
| shimmer | female-tianmei | Female voice |

You can also use any MiniMax-native voice ID directly by passing it as the `voice` parameter.


### Streaming (WebSocket)

:::note
The current implementation uses MiniMax's HTTP endpoint. For WebSocket streaming support, please refer to MiniMax's official documentation at [https://platform.minimax.io/docs](https://platform.minimax.io/docs).
:::

## Error Handling

```python
from litellm import speech
import litellm

try:
    response = speech(
        model="minimax/speech-2.6-hd",
        voice="alloy",
        input="Test input",
    )
    response.stream_to_file("output.mp3")
except litellm.exceptions.BadRequestError as e:
    print(f"Bad request: {e}")
except litellm.exceptions.AuthenticationError as e:
    print(f"Authentication failed: {e}")
except Exception as e:
    print(f"Error: {e}")
```

### Extra Body Parameters

Pass these via `extra_body`:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| vol | float | Volume (0.1 to 10) | 1.0 |
| pitch | int | Pitch adjustment (-12 to 12) | 0 |
| sample_rate | int | Sample rate: 16000, 24000, 32000 | 32000 |
| bitrate | int | Bitrate for MP3: 64000, 128000, 192000, 256000 | 128000 |
| channel | int | Audio channels: 1 (mono) or 2 (stereo) | 1 |
| output_format | string | Output format: "hex" or "url" (url returns a URL valid for 24 hours) | hex |
