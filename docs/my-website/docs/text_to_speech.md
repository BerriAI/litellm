import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /audio/speech

## **LiteLLM Python SDK Usage**
### Quick Start 

```python
from pathlib import Path
from litellm import speech
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

speech_file_path = Path(__file__).parent / "speech.mp3"
response = speech(
        model="openai/tts-1",
        voice="alloy",
        input="the quick brown fox jumped over the lazy dogs",
    )
response.stream_to_file(speech_file_path)
```

### Async Usage 

```python
from litellm import aspeech
from pathlib import Path
import os, asyncio

os.environ["OPENAI_API_KEY"] = "sk-.."

async def test_async_speech(): 
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = await litellm.aspeech(
            model="openai/tts-1",
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=None,
            api_key=None,
            organization=None,
            project=None,
            max_retries=1,
            timeout=600,
            client=None,
            optional_params={},
        )
    response.stream_to_file(speech_file_path)

asyncio.run(test_async_speech())
```

## **LiteLLM Proxy Usage**

LiteLLM provides an openai-compatible `/audio/speech` endpoint for Text-to-speech calls.

```bash
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "alloy"
  }' \
  --output speech.mp3
```

**Setup**

```bash
- model_name: tts
  litellm_params:
    model: openai/tts-1
    api_key: os.environ/OPENAI_API_KEY
```

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```
## **Supported Providers**

| Provider    | Link to Usage      |
|-------------|--------------------|
| OpenAI      |   [Usage](#quick-start)                 |
| Azure OpenAI|   [Usage](../docs/providers/azure#azure-text-to-speech-tts)                 |
| Vertex AI   |   [Usage](../docs/providers/vertex#text-to-speech-apis)                 |
| Gemini      |   [Usage](#gemini-text-to-speech)                 |
| ElevenLabs  |   [Usage](#elevenlabs-text-to-speech)             |

## `/audio/speech` to `/chat/completions` Bridge

LiteLLM allows you to use `/chat/completions` models to generate speech through the `/audio/speech` endpoint. This is useful for models like Gemini's TTS-enabled models that are only accessible via `/chat/completions`.

### Gemini Text-to-Speech

#### Python SDK Usage

```python showLineNumbers title="Gemini Text-to-Speech SDK Usage"
import litellm
import os

# Set your Gemini API key
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

def test_audio_speech_gemini():
    result = litellm.speech(
        model="gemini/gemini-2.5-flash-preview-tts",
        input="the quick brown fox jumped over the lazy dogs",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Save to file
    from pathlib import Path
    speech_file_path = Path(__file__).parent / "gemini_speech.mp3"
    result.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

test_audio_speech_gemini()
```

#### Async Usage

```python showLineNumbers title="Gemini Text-to-Speech Async Usage"
import litellm
import asyncio
import os
from pathlib import Path

os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

async def test_async_gemini_speech():
    speech_file_path = Path(__file__).parent / "gemini_speech.mp3"
    response = await litellm.aspeech(
        model="gemini/gemini-2.5-flash-preview-tts",
        input="the quick brown fox jumped over the lazy dogs",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    response.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

asyncio.run(test_async_gemini_speech())
```

#### LiteLLM Proxy Usage

**Setup Config:**

```yaml showLineNumbers title="Gemini Proxy Configuration"
model_list:
- model_name: gemini-tts
  litellm_params:
    model: gemini/gemini-2.5-flash-preview-tts
    api_key: os.environ/GEMINI_API_KEY
```

**Start Proxy:**

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**Make Request:**

```bash showLineNumbers title="Gemini TTS Request"
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-tts",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "alloy"
  }' \
  --output gemini_speech.mp3
```

### Vertex AI Text-to-Speech

#### Python SDK Usage

```python showLineNumbers title="Vertex AI Text-to-Speech SDK Usage"
import litellm
import os
from pathlib import Path

# Set your Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/service-account.json"

def test_audio_speech_vertex():
    result = litellm.speech(
        model="vertex_ai/gemini-2.5-flash-preview-tts",
        input="the quick brown fox jumped over the lazy dogs",
    )
    
    # Save to file
    speech_file_path = Path(__file__).parent / "vertex_speech.mp3"
    result.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

test_audio_speech_vertex()
```

#### LiteLLM Proxy Usage

**Setup Config:**

```yaml showLineNumbers title="Vertex AI Proxy Configuration"
model_list:
- model_name: vertex-tts
  litellm_params:
    model: vertex_ai/gemini-2.5-flash-preview-tts
    vertex_project: your-project-id
    vertex_location: us-central1
```

**Make Request:**

```bash showLineNumbers title="Vertex AI TTS Request"
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vertex-tts",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "en-US-Wavenet-D"
  }' \
  --output vertex_speech.mp3
```

### ElevenLabs Text-to-Speech

#### Python SDK Usage

```python showLineNumbers title="ElevenLabs Text-to-Speech SDK Usage"
import litellm
import os
from pathlib import Path

# Set your ElevenLabs API key
os.environ["ELEVENLABS_API_KEY"] = "your-elevenlabs-api-key"

def test_audio_speech_elevenlabs():
    result = litellm.speech(
        model="elevenlabs/eleven_multilingual_v2",
        input="the quick brown fox jumped over the lazy dogs",
        voice="21m00Tcm4TlvDq8ikWAM",  # ElevenLabs voice ID
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    
    # Save to file
    speech_file_path = Path(__file__).parent / "elevenlabs_speech.mp3"
    result.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

test_audio_speech_elevenlabs()
```

#### Async Usage

```python showLineNumbers title="ElevenLabs Text-to-Speech Async Usage"
import litellm
import asyncio
import os
from pathlib import Path

os.environ["ELEVENLABS_API_KEY"] = "your-elevenlabs-api-key"

async def test_async_elevenlabs_speech():
    speech_file_path = Path(__file__).parent / "elevenlabs_speech.mp3"
    response = await litellm.aspeech(
        model="elevenlabs/eleven_multilingual_v2",
        input="the quick brown fox jumped over the lazy dogs",
        voice="21m00Tcm4TlvDq8ikWAM",
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    response.stream_to_file(speech_file_path)
    print(f"Audio saved to {speech_file_path}")

asyncio.run(test_async_elevenlabs_speech())
```

#### LiteLLM Proxy Usage

**Setup Config:**

```yaml showLineNumbers title="ElevenLabs Proxy Configuration"
model_list:
- model_name: elevenlabs-tts
  litellm_params:
    model: elevenlabs/eleven_multilingual_v2
    api_key: os.environ/ELEVENLABS_API_KEY
```

**Start Proxy:**

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**Make Request:**

```bash showLineNumbers title="ElevenLabs TTS Request"
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "elevenlabs-tts",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "21m00Tcm4TlvDq8ikWAM"
  }' \
  --output elevenlabs_speech.mp3
```

#### Supported Models

| Model | Description |
|-------|-------------|
| eleven_multilingual_v2 | High-quality TTS with support for 29 languages |
| eleven_flash_v2_5 | Fast, affordable TTS with ultra-low latency |
| eleven_turbo_v2_5 | Balanced quality and speed with low latency |

#### Advanced Features

ElevenLabs supports additional parameters for fine-tuning voice output:

```python showLineNumbers title="ElevenLabs Advanced Usage"
import litellm

response = litellm.speech(
    model="elevenlabs/eleven_multilingual_v2",
    input="Hello, this is a test with custom voice settings",
    voice="21m00Tcm4TlvDq8ikWAM",
    # ElevenLabs-specific parameters
    voice_settings={
        "stability": 0.5,
        "similarity_boost": 0.8,
        "style": 0.0,
        "use_speaker_boost": True
    },
    # You can also specify output format
    response_format="mp3",  # maps to mp3_44100_128
)
```

## ✨ Enterprise LiteLLM Proxy - Set Max Request File Size 

Use this when you want to limit the file size for requests sent to `audio/transcriptions`

```yaml
- model_name: whisper
  litellm_params:
    model: whisper-1
    api_key: sk-*******
    max_file_size_mb: 0.00001 # 👈 max file size in MB  (Set this intentionally very small for testing)
  model_info:
    mode: audio_transcription
```

Make a test Request with a valid file
```shell
curl --location 'http://localhost:4000/v1/audio/transcriptions' \
--header 'Authorization: Bearer sk-1234' \
--form 'file=@"/Users/ishaanjaffer/Github/litellm/tests/gettysburg.wav"' \
--form 'model="whisper"'
```


Expect to see the follow response 

```shell
{"error":{"message":"File size is too large. Please check your file size. Passed file size: 0.7392807006835938 MB. Max file size: 0.0001 MB","type":"bad_request","param":"file","code":500}}%  
```