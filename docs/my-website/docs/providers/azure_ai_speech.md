# Azure AI Speech (Cognitive Services)

Azure AI Speech is Azure's Cognitive Services text-to-speech API, separate from Azure OpenAI. It provides high-quality neural voices with broader language support and advanced speech customization.

**When to use this vs Azure OpenAI TTS:**
- **Azure AI Speech** - More languages, neural voices, SSML support, speech customization
- **Azure OpenAI TTS** - OpenAI models, integrated with Azure OpenAI services


## Overview

| Property | Details |
|-------|-------|
| Description | Azure AI Speech is Azure's Cognitive Services text-to-speech API, separate from Azure OpenAI. It provides high-quality neural voices with broader language support and advanced speech customization. |
| Provider Route on LiteLLM | `azure/speech/` |

## Quick Start

**LiteLLM SDK**

```python showLineNumbers title="SDK Usage"
from litellm import speech
from pathlib import Path
import os

os.environ["AZURE_TTS_API_KEY"] = "your-cognitive-services-key"

speech_file_path = Path(__file__).parent / "speech.mp3"
response = speech(
    model="azure/speech/azure-tts",
    voice="alloy",
    input="Hello, this is Azure AI Speech",
    api_base="https://eastus.tts.speech.microsoft.com",
    api_key=os.environ["AZURE_TTS_API_KEY"],
)
response.stream_to_file(speech_file_path)
```

**LiteLLM Proxy**

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: azure-speech
    litellm_params:
      model: azure/speech/azure-tts
      api_base: https://eastus.tts.speech.microsoft.com
      api_key: os.environ/AZURE_TTS_API_KEY
```

## Setup

1. Create an Azure Cognitive Services resource in the [Azure Portal](https://portal.azure.com)
2. Get your API key from the resource
3. Note your region (e.g., `eastus`, `westus`, `westeurope`)
4. Use the regional endpoint: `https://{region}.tts.speech.microsoft.com`

## Voice Mapping

LiteLLM automatically maps OpenAI voice names to Azure Neural voices:

| OpenAI Voice | Azure Neural Voice | Description |
|-------------|-------------------|-------------|
| `alloy` | en-US-JennyNeural | Neutral and balanced |
| `echo` | en-US-GuyNeural | Warm and upbeat |
| `fable` | en-GB-RyanNeural | Expressive and dramatic |
| `onyx` | en-US-DavisNeural | Deep and authoritative |
| `nova` | en-US-AmberNeural | Friendly and conversational |
| `shimmer` | en-US-AriaNeural | Bright and cheerful |

## Supported Parameters

```python showLineNumbers title="All Parameters"
response = speech(
    model="azure/speech/azure-tts",
    voice="alloy",                    # Required: Voice selection
    input="text to convert",          # Required: Input text
    speed=1.0,                        # Optional: 0.25 to 4.0 (default: 1.0)
    response_format="mp3",            # Optional: mp3, opus, wav, pcm
    api_base="https://eastus.tts.speech.microsoft.com",
    api_key="your-key",
)
```

### Response Formats

| Format | Azure Output Format | Sample Rate |
|--------|-------------------|-------------|
| `mp3` | audio-24khz-48kbitrate-mono-mp3 | 24kHz |
| `opus` | ogg-48khz-16bit-mono-opus | 48kHz |
| `wav` | riff-24khz-16bit-mono-pcm | 24kHz |
| `pcm` | raw-24khz-16bit-mono-pcm | 24kHz |

## Async Support

```python showLineNumbers title="Async Usage"
import asyncio
from litellm import aspeech
from pathlib import Path

async def generate_speech():
    response = await aspeech(
        model="azure/speech/azure-tts",
        voice="alloy",
        input="Hello from async",
        api_base="https://eastus.tts.speech.microsoft.com",
        api_key=os.environ["AZURE_TTS_API_KEY"],
    )
    
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response.stream_to_file(speech_file_path)

asyncio.run(generate_speech())
```

## Regional Endpoints

Replace `{region}` with your Azure resource region:

- US East: `https://eastus.tts.speech.microsoft.com`
- US West: `https://westus.tts.speech.microsoft.com`
- Europe West: `https://westeurope.tts.speech.microsoft.com`
- Asia Southeast: `https://southeastasia.tts.speech.microsoft.com`

[Full list of regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions)

## Advanced Features

### Custom Neural Voices

You can use any Azure Neural voice by passing the full voice name:

```python showLineNumbers title="Custom Voice"
response = speech(
    model="azure/speech/azure-tts",
    voice="en-US-AriaNeural",  # Direct Azure voice name
    input="Using a specific neural voice",
    api_base="https://eastus.tts.speech.microsoft.com",
    api_key=os.environ["AZURE_TTS_API_KEY"],
)
```

Browse available voices in the [Azure Speech Gallery](https://speech.microsoft.com/portal/voicegallery).

## Error Handling

```python showLineNumbers title="Error Handling"
from litellm import speech
from litellm.exceptions import APIError

try:
    response = speech(
        model="azure/speech/azure-tts",
        voice="alloy",
        input="Test message",
        api_base="https://eastus.tts.speech.microsoft.com",
        api_key=os.environ["AZURE_TTS_API_KEY"],
    )
except APIError as e:
    print(f"Azure Speech error: {e}")
```

## Reference

- [Azure Speech Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/)
- [Text-to-Speech REST API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech)

