import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MiniMax - Text-to-Speech

## Overview

MiniMax provides high-quality text-to-speech synthesis with support for 40+ languages and ultra-low latency. LiteLLM provides a unified OpenAI-compatible interface for MiniMax TTS.

| Feature | Supported | Notes |
|---------|-----------|-------|
| Logging | ✅ | Works across all integrations |
| Fallbacks | ✅ | Works between supported models |
| Loadbalancing | ✅ | Works between supported models |
| Guardrails | ✅ | Applies to input text |
| Supported Models | speech-2.6-hd, speech-2.6-turbo, speech-02-hd, speech-02-turbo | |

## Supported Models

| Model | Description |
|-------|-------------|
| speech-2.6-hd | Ultra-low latency, intelligence parsing, and enhanced naturalness |
| speech-2.6-turbo | Faster, more affordable, ideal for agents |
| speech-02-hd | Superior rhythm and stability with outstanding replication similarity |
| speech-02-turbo | Superior rhythm and stability with enhanced multilingual capabilities |
| speech-01-hd | Previous generation HD model |
| speech-01-turbo | Previous generation turbo model |

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
