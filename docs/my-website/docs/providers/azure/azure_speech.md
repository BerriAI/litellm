# Azure Text to Speech (tts)

## Overview

| Property | Details |
|-------|-------|
| Description | Convert text to natural-sounding speech using Azure OpenAI's Text to Speech models |
| Provider Route on LiteLLM | `azure/` |
| Supported Operations | `/audio/speech` |
| Link to Provider Doc | [Azure OpenAI TTS ↗](https://learn.microsoft.com/en-us/azure/ai-services/openai/text-to-speech-quickstart)

## Quick Start

### **LiteLLM SDK**

```python showLineNumbers title="SDK Usage"
from litellm import speech
from pathlib import Path
import os

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# azure call
speech_file_path = Path(__file__).parent / "speech.mp3"
response = speech(
        model="azure/<your-deployment-name>",
        voice="alloy",
        input="the quick brown fox jumped over the lazy dogs",
    )
response.stream_to_file(speech_file_path)
```

### **LiteLLM PROXY**

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
 - model_name: azure/tts-1
    litellm_params:
      model: azure/tts-1
      api_base: "os.environ/AZURE_API_BASE_TTS"
      api_key: "os.environ/AZURE_API_KEY_TTS"
      api_version: "os.environ/AZURE_API_VERSION" 
```

## Available Voices

Azure OpenAI supports the following voices:
- `alloy` - Neutral and balanced
- `echo` - Warm and upbeat
- `fable` - Expressive and dramatic
- `onyx` - Deep and authoritative
- `nova` - Friendly and conversational
- `shimmer` - Bright and cheerful

## Supported Parameters

```python showLineNumbers title="All Parameters"
response = speech(
    model="azure/<your-deployment-name>",
    voice="alloy",                    # Required: Voice selection
    input="text to convert",          # Required: Input text
    speed=1.0,                        # Optional: 0.25 to 4.0 (default: 1.0)
    response_format="mp3"             # Optional: mp3, opus, aac, flac, wav, pcm
)
```

## Supported Models

- `tts-1` - Standard quality, optimized for speed
- `tts-1-hd` - High definition, optimized for quality

Use your Azure deployment name: `azure/<your-deployment-name>`