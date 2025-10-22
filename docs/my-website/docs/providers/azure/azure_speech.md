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

### OpenAI-Compatible Parameters

```python showLineNumbers title="Standard Parameters"
response = speech(
    model="azure/<your-deployment-name>",
    voice="alloy",                    # Required: Voice selection
    input="text to convert",          # Required: Input text
    speed=1.0,                        # Optional: 0.25 to 4.0 (default: 1.0)
    response_format="mp3"             # Optional: mp3, opus, aac, flac, wav, pcm
)
```

## Sending Azure-Specific Params

Azure TTS supports additional SSML features through optional parameters:

- style: Speaking style (e.g., "cheerful", "sad", "angry")
- styledegree: Style intensity (0.01 to 2)
- role: Voice role (e.g., "Girl", "Boy", "SeniorFemale", "SeniorMale")
- lang: Language code for multilingual voices (e.g., "es-ES", "fr-FR")


#### **Using LiteLLM AI Gateway (CURL)**

#### Custom Azure Voice

```bash
curl -X POST 'http://0.0.0.0:4000/v1/audio/speech' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "azure-tts",
    "voice": "en-US-AndrewNeural",
    "input": "Hello, this is a test"
  }' \
  --output speech.mp3
```

#### Speaking Style

```bash
curl -X POST 'http://0.0.0.0:4000/v1/audio/speech' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "azure-tts",
    "voice": "en-US-AriaNeural",
    "input": "That would be just amazing!",
    "style": "cheerful"
  }' \
  --output speech.mp3
```

#### Style with Degree and Role

```bash
curl -X POST 'http://0.0.0.0:4000/v1/audio/speech' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "azure-tts",
    "voice": "en-US-AriaNeural",
    "input": "Good morning! How are you today?",
    "style": "cheerful",
    "styledegree": "2",
    "role": "SeniorFemale"
  }' \
  --output speech.mp3
```

#### Language Override

```bash
curl -X POST 'http://0.0.0.0:4000/v1/audio/speech' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "azure-tts",
    "voice": "en-US-AvaMultilingualNeural",
    "input": "Hola, ¿cómo estás?",
    "lang": "es-ES"
  }' \
  --output speech.mp3
```

## Azure-Specific Parameters Reference

| Parameter | Description | Example Values | Notes |
|-----------|-------------|----------------|-------|
| `style` | Speaking style | `cheerful`, `sad`, `angry`, `excited`, `friendly`, `hopeful`, `shouting`, `terrified`, `unfriendly`, `whispering` | Only supported by certain voices. See [Azure voice styles documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice#use-speaking-styles-and-roles) |
| `styledegree` | Style intensity | `0.01` to `2` | Higher values = more intense. Default is `1` |
| `role` | Voice role | `Girl`, `Boy`, `YoungAdultFemale`, `YoungAdultMale`, `OlderAdultFemale`, `OlderAdultMale`, `SeniorFemale`, `SeniorMale` | Only supported by certain voices |
| `lang` | Language code | `es-ES`, `fr-FR`, `de-DE`, etc. | For multilingual voices. Overrides the default language |

## Supported Models

- `tts-1` - Standard quality, optimized for speed
- `tts-1-hd` - High definition, optimized for quality

Use your Azure deployment name: `azure/<your-deployment-name>`