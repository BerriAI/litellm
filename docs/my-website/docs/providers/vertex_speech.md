import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vertex AI Text to Speech

| Property | Details |
|-------|-------|
| Description | Google Cloud Text-to-Speech with Chirp3 HD voices and Gemini TTS |
| Provider Route on LiteLLM | `vertex_ai/` (Chirp), `vertex_ai/gemini-*-tts` (Gemini) |

## Chirp3 HD Voices

Google Cloud Text-to-Speech API with high-quality Chirp3 HD voices.

### Quick Start

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import speech
from pathlib import Path

speech_file_path = Path(__file__).parent / "speech.mp3"
response = speech(
    model="vertex_ai/",
    voice="alloy",  # OpenAI voice name - automatically mapped
    input="Hello, this is Vertex AI Text to Speech",
    vertex_project="your-project-id",
    vertex_location="us-central1",
)
response.stream_to_file(speech_file_path)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```yaml
model_list:
  - model_name: vertex-tts
    litellm_params:
      model: vertex_ai/
      vertex_project: "your-project-id"
      vertex_location: "us-central1"
      vertex_credentials: "/path/to/service_account.json"
```

```bash
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vertex-tts",
    "voice": "alloy",
    "input": "Hello, this is Vertex AI Text to Speech"
  }' \
  --output speech.mp3
```

</TabItem>
</Tabs>

### Voice Mapping

LiteLLM maps OpenAI voice names to Google Cloud voices. You can use either OpenAI voices or Google Cloud voices directly.

| OpenAI Voice | Google Cloud Voice |
|-------------|-------------------|
| `alloy` | en-US-Studio-O |
| `echo` | en-US-Studio-M |
| `fable` | en-GB-Studio-B |
| `onyx` | en-US-Wavenet-D |
| `nova` | en-US-Studio-O |
| `shimmer` | en-US-Wavenet-F |

### Using Google Cloud Voices Directly

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import speech
from pathlib import Path

# Pass Google Cloud voice name directly
response = speech(
    model="vertex_ai/",
    voice="en-US-Neural2-A",
    input="Hello with a Neural2 voice",
    vertex_project="your-project-id",
)
response.stream_to_file("speech.mp3")

# Or pass as dict for full control
response = speech(
    model="vertex_ai/",
    voice={
        "languageCode": "de-DE",
        "name": "de-DE-Wavenet-A",
    },
    input="Hallo, dies ist ein Test",
    vertex_project="your-project-id",
)
response.stream_to_file("speech.mp3")
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```bash
# Pass Google Cloud voice name directly
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vertex-tts",
    "voice": "en-US-Neural2-A",
    "input": "Hello with a Neural2 voice"
  }' \
  --output speech.mp3

# Or pass as dict
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vertex-tts",
    "voice": {"languageCode": "de-DE", "name": "de-DE-Wavenet-A"},
    "input": "Hallo, dies ist ein Test"
  }' \
  --output speech.mp3
```

</TabItem>
</Tabs>

Browse available voices: [Google Cloud Text-to-Speech Console](https://console.cloud.google.com/vertex-ai/generative/speech/text-to-speech)

### Passing Raw SSML

LiteLLM auto-detects SSML when your input contains `<speak>` tags and passes it through unchanged.

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import speech
from pathlib import Path

ssml = """
<speak>
    <p>Hello, world!</p>
    <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
</speak>
"""

response = speech(
    model="vertex_ai/",
    voice="en-US-Studio-O",
    input=ssml,  # Auto-detected as SSML
    vertex_project="your-project-id",
)
response.stream_to_file("speech.mp3")

# Force SSML mode with use_ssml=True
response = speech(
    model="vertex_ai/",
    voice="en-US-Studio-O",
    input="<speak><prosody rate='slow'>Speaking slowly</prosody></speak>",
    use_ssml=True,
    vertex_project="your-project-id",
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```bash
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vertex-tts",
    "voice": "en-US-Studio-O",
    "input": "<speak><p>Hello!</p><break time=\"500ms\"/><p>How are you?</p></speak>"
  }' \
  --output speech.mp3
```

</TabItem>
</Tabs>

### Supported Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| `voice` | Voice selection | OpenAI voice, Google Cloud voice name, or dict |
| `input` | Text to convert | Plain text or SSML |
| `speed` | Speaking rate | 0.25 to 4.0 (default: 1.0) |
| `response_format` | Audio format | `mp3`, `opus`, `wav`, `pcm`, `flac` |
| `use_ssml` | Force SSML mode | `True` / `False` |

### Async Usage

```python
import asyncio
from litellm import aspeech

async def main():
    response = await aspeech(
        model="vertex_ai/",
        voice="alloy",
        input="Hello from async",
        vertex_project="your-project-id",
    )
    response.stream_to_file("speech.mp3")

asyncio.run(main())
```

---

## Gemini TTS

Gemini models with audio output capabilities using the chat completions API.

:::warning
**Limitations:**
- Only supports `pcm16` audio format
- Streaming not yet supported
- Must set `modalities: ["audio"]`
:::

### Quick Start

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import completion
import json

# Load credentials
with open('path/to/service_account.json', 'r') as file:
    vertex_credentials = json.dumps(json.load(file))

response = completion(
    model="vertex_ai/gemini-2.5-flash-preview-tts",
    messages=[{"role": "user", "content": "Say hello in a friendly voice"}],
    modalities=["audio"],
    audio={
        "voice": "Kore",
        "format": "pcm16"
    },
    vertex_credentials=vertex_credentials
)
print(response)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```yaml
model_list:
  - model_name: gemini-tts
    litellm_params:
      model: vertex_ai/gemini-2.5-flash-preview-tts
      vertex_project: "your-project-id"
      vertex_location: "us-central1"
      vertex_credentials: "/path/to/service_account.json"
```

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gemini-tts",
    "messages": [{"role": "user", "content": "Say hello in a friendly voice"}],
    "modalities": ["audio"],
    "audio": {"voice": "Kore", "format": "pcm16"}
  }'
```

</TabItem>
</Tabs>

### Supported Models

- `vertex_ai/gemini-2.5-flash-preview-tts`
- `vertex_ai/gemini-2.5-pro-preview-tts`

See [Gemini TTS documentation](https://ai.google.dev/gemini-api/docs/speech-generation) for available voices.

### Advanced Usage

```python
response = completion(
    model="vertex_ai/gemini-2.5-pro-preview-tts",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that speaks clearly."},
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    modalities=["audio"],
    audio={"voice": "Charon", "format": "pcm16"},
    temperature=0.7,
    max_tokens=150,
    vertex_credentials=vertex_credentials
)
```
