import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vertex AI Text to Speech

| Property | Details |
|-------|-------|
| Description | Google Cloud Text-to-Speech with Chirp3 HD voices and Gemini TTS |
| Provider Route on LiteLLM | `vertex_ai/chirp` (Chirp), `vertex_ai/gemini-*-tts` (Gemini) |

## Chirp3 HD Voices

Google Cloud Text-to-Speech API with high-quality Chirp3 HD voices.

### Quick Start

#### LiteLLM Python SDK

```python showLineNumbers title="Chirp3 Quick Start"
from litellm import speech
from pathlib import Path

speech_file_path = Path(__file__).parent / "speech.mp3"
response = speech(
    model="vertex_ai/chirp",
    voice="alloy",  # OpenAI voice name - automatically mapped
    input="Hello, this is Vertex AI Text to Speech",
    vertex_project="your-project-id",
    vertex_location="us-central1",
)
response.stream_to_file(speech_file_path)
```

#### LiteLLM AI Gateway

**1. Setup config.yaml**

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: vertex-tts
    litellm_params:
      model: vertex_ai/chirp
      vertex_project: "your-project-id"
      vertex_location: "us-central1"
      vertex_credentials: "/path/to/service_account.json"
```

**2. Start the proxy**

```bash title="Start LiteLLM Proxy"
litellm --config /path/to/config.yaml
```

**3. Make requests**

<Tabs>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="Chirp3 Quick Start"
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
<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="Chirp3 Quick Start"
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.audio.speech.create(
    model="vertex-tts",
    voice="alloy",
    input="Hello, this is Vertex AI Text to Speech",
)
response.stream_to_file("speech.mp3")
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

#### LiteLLM Python SDK

```python showLineNumbers title="Chirp3 HD Voice"
from litellm import speech

# Pass Chirp3 HD voice name directly
response = speech(
    model="vertex_ai/chirp",
    voice="en-US-Chirp3-HD-Charon",
    input="Hello with a Chirp3 HD voice",
    vertex_project="your-project-id",
)
response.stream_to_file("speech.mp3")
```

```python showLineNumbers title="Voice as Dict (Multilingual)"
from litellm import speech

# Pass as dict for full control over language and voice
response = speech(
    model="vertex_ai/chirp",
    voice={
        "languageCode": "de-DE",
        "name": "de-DE-Chirp3-HD-Charon",
    },
    input="Hallo, dies ist ein Test",
    vertex_project="your-project-id",
)
response.stream_to_file("speech.mp3")
```

#### LiteLLM AI Gateway

<Tabs>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="Chirp3 HD Voice"
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vertex-tts",
    "voice": "en-US-Chirp3-HD-Charon",
    "input": "Hello with a Chirp3 HD voice"
  }' \
  --output speech.mp3
```

```bash showLineNumbers title="Voice as Dict (Multilingual)"
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vertex-tts",
    "voice": {"languageCode": "de-DE", "name": "de-DE-Chirp3-HD-Charon"},
    "input": "Hallo, dies ist ein Test"
  }' \
  --output speech.mp3
```

</TabItem>
<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="Chirp3 HD Voice"
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.audio.speech.create(
    model="vertex-tts",
    voice="en-US-Chirp3-HD-Charon",
    input="Hello with a Chirp3 HD voice",
)
response.stream_to_file("speech.mp3")
```

```python showLineNumbers title="Voice as Dict (Multilingual)"
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.audio.speech.create(
    model="vertex-tts",
    voice={"languageCode": "de-DE", "name": "de-DE-Chirp3-HD-Charon"},
    input="Hallo, dies ist ein Test",
)
response.stream_to_file("speech.mp3")
```

</TabItem>
</Tabs>

Browse available voices: [Google Cloud Text-to-Speech Console](https://console.cloud.google.com/vertex-ai/generative/speech/text-to-speech)

### Passing Raw SSML

LiteLLM auto-detects SSML when your input contains `<speak>` tags and passes it through unchanged.

#### LiteLLM Python SDK

```python showLineNumbers title="SSML Input"
from litellm import speech

ssml = """
<speak>
    <p>Hello, world!</p>
    <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
</speak>
"""

response = speech(
    model="vertex_ai/chirp",
    voice="en-US-Studio-O",
    input=ssml,  # Auto-detected as SSML
    vertex_project="your-project-id",
)
response.stream_to_file("speech.mp3")
```

```python showLineNumbers title="Force SSML Mode"
from litellm import speech

# Force SSML mode with use_ssml=True
response = speech(
    model="vertex_ai/chirp",
    voice="en-US-Studio-O",
    input="<speak><prosody rate='slow'>Speaking slowly</prosody></speak>",
    use_ssml=True,
    vertex_project="your-project-id",
)
response.stream_to_file("speech.mp3")
```

#### LiteLLM AI Gateway

<Tabs>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="SSML Input"
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
<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="SSML Input"
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

ssml = """<speak><p>Hello!</p><break time="500ms"/><p>How are you?</p></speak>"""

response = client.audio.speech.create(
    model="vertex-tts",
    voice="en-US-Studio-O",
    input=ssml,
)
response.stream_to_file("speech.mp3")
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

```python showLineNumbers title="Async Speech Generation"
import asyncio
from litellm import aspeech

async def main():
    response = await aspeech(
        model="vertex_ai/chirp",
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

#### LiteLLM Python SDK

```python showLineNumbers title="Gemini TTS Quick Start"
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

#### LiteLLM AI Gateway

**1. Setup config.yaml**

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gemini-tts
    litellm_params:
      model: vertex_ai/gemini-2.5-flash-preview-tts
      vertex_project: "your-project-id"
      vertex_location: "us-central1"
      vertex_credentials: "/path/to/service_account.json"
```

**2. Start the proxy**

```bash title="Start LiteLLM Proxy"
litellm --config /path/to/config.yaml
```

**3. Make requests**

<Tabs>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="Gemini TTS Request"
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
<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="Gemini TTS Request"
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.chat.completions.create(
    model="gemini-tts",
    messages=[{"role": "user", "content": "Say hello in a friendly voice"}],
    modalities=["audio"],
    audio={"voice": "Kore", "format": "pcm16"},
)
print(response)
```

</TabItem>
</Tabs>

### Supported Models

- `vertex_ai/gemini-2.5-flash-preview-tts`
- `vertex_ai/gemini-2.5-pro-preview-tts`

See [Gemini TTS documentation](https://ai.google.dev/gemini-api/docs/speech-generation) for available voices.

### Advanced Usage

```python showLineNumbers title="Gemini TTS with System Prompt"
from litellm import completion

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
