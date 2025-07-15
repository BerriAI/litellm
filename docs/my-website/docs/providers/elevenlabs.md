import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ElevenLabs

ElevenLabs provides high-quality AI voice technology, including speech-to-text capabilities through their transcription API.

| Property | Details |
|----------|---------|
| Description | ElevenLabs offers advanced AI voice technology with speech-to-text transcription capabilities that support multiple languages and speaker diarization. |
| Provider Route on LiteLLM | `elevenlabs/` |
| Provider Doc | [ElevenLabs API ↗](https://elevenlabs.io/docs/api-reference) |
| Supported Endpoints | `/audio/transcriptions`, `/audio/speech` |

## Quick Start

### LiteLLM Python SDK

<Tabs>
<TabItem value="basic" label="Basic Usage">

```python showLineNumbers title="Basic audio transcription with ElevenLabs"
import litellm

# Transcribe audio file
with open("audio.mp3", "rb") as audio_file:
    response = litellm.transcription(
        model="elevenlabs/scribe_v1",
        file=audio_file,
        api_key="your-elevenlabs-api-key"  # or set ELEVENLABS_API_KEY env var
    )

print(response.text)
```

</TabItem>

<TabItem value="advanced" label="Advanced Features">

```python showLineNumbers title="Audio transcription with advanced features"
import litellm

# Transcribe with speaker diarization and language specification
with open("audio.wav", "rb") as audio_file:
    response = litellm.transcription(
        model="elevenlabs/scribe_v1",
        file=audio_file,
        language="en",           # Language hint (maps to language_code)
        temperature=0.3,         # Control randomness in transcription
        diarize=True,           # Enable speaker diarization
        api_key="your-elevenlabs-api-key"
    )

print(f"Transcription: {response.text}")
print(f"Language: {response.language}")

# Access word-level timestamps if available
if hasattr(response, 'words') and response.words:
    for word_info in response.words:
        print(f"Word: {word_info['word']}, Start: {word_info['start']}, End: {word_info['end']}")
```

</TabItem>

<TabItem value="async" label="Async Usage">

```python showLineNumbers title="Async audio transcription"
import litellm
import asyncio

async def transcribe_audio():
    with open("audio.mp3", "rb") as audio_file:
        response = await litellm.atranscription(
            model="elevenlabs/scribe_v1",
            file=audio_file,
            api_key="your-elevenlabs-api-key"
        )
    
    return response.text

# Run async transcription
result = asyncio.run(transcribe_audio())
print(result)
```

</TabItem>
</Tabs>

### LiteLLM Proxy

#### 1. Configure your proxy

<Tabs>
<TabItem value="config-yaml" label="config.yaml">

```yaml showLineNumbers title="ElevenLabs configuration in config.yaml"
model_list:
  - model_name: elevenlabs-transcription
    litellm_params:
      model: elevenlabs/scribe_v1
      api_key: os.environ/ELEVENLABS_API_KEY

general_settings:
  master_key: your-master-key
```

</TabItem>

<TabItem value="env-vars" label="Environment Variables">

```bash showLineNumbers title="Required environment variables"
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
export LITELLM_MASTER_KEY="your-master-key"
```

</TabItem>
</Tabs>

#### 2. Start the proxy

```bash showLineNumbers title="Start LiteLLM proxy server"
litellm --config config.yaml

# Proxy will be available at http://localhost:4000
```

#### 3. Make transcription requests

<Tabs>
<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Audio transcription with curl"
curl http://localhost:4000/v1/audio/transcriptions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@audio.mp3" \
  -F model="elevenlabs-transcription" \
  -F language="en" \
  -F temperature="0.3"
```

</TabItem>

<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="Using OpenAI SDK with LiteLLM proxy"
from openai import OpenAI

# Initialize client with your LiteLLM proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-api-key"
)

# Transcribe audio file
with open("audio.mp3", "rb") as audio_file:
    response = client.audio.transcriptions.create(
        model="elevenlabs-transcription",
        file=audio_file,
        language="en",
        temperature=0.3,
        # ElevenLabs-specific parameters
        diarize=True,
        speaker_boost=True,
        custom_vocabulary="technical,AI,machine learning"
    )

print(response.text)
```

</TabItem>

<TabItem value="javascript" label="JavaScript/Node.js">

```javascript showLineNumbers title="Audio transcription with JavaScript"
import OpenAI from 'openai';
import fs from 'fs';

const openai = new OpenAI({
  baseURL: 'http://localhost:4000',
  apiKey: 'your-litellm-api-key'
});

async function transcribeAudio() {
  const response = await openai.audio.transcriptions.create({
    file: fs.createReadStream('audio.mp3'),
    model: 'elevenlabs-transcription',
    language: 'en',
    temperature: 0.3,
    diarize: true,
    speaker_boost: true
  });

  console.log(response.text);
}

transcribeAudio();
```

</TabItem>
</Tabs>

## Response Format

ElevenLabs returns transcription responses in OpenAI-compatible format:

```json showLineNumbers title="Example transcription response"
{
  "text": "Hello, this is a sample transcription with multiple speakers.",
  "task": "transcribe",
  "language": "en",
  "words": [
    {
      "word": "Hello",
      "start": 0.0,
      "end": 0.5
    },
    {
      "word": "this",
      "start": 0.5,
      "end": 0.8
    }
  ]
}
```

### Common Issues

1. **Invalid API Key**: Ensure `ELEVENLABS_API_KEY` is set correctly

## Text-to-Speech

ElevenLabs also provides high-quality text-to-speech capabilities through their TTS API.

### LiteLLM Python SDK

<Tabs>
<TabItem value="basic" label="Basic Usage">

```python showLineNumbers title="Basic text-to-speech with ElevenLabs"
import litellm
from pathlib import Path

speech_file_path = Path(__file__).parent / "speech.mp3"
response = litellm.speech(
    model="elevenlabs/eleven_multilingual_v2",
    input="Hello, this is a test of ElevenLabs text-to-speech",
    voice="21m00Tcm4TlvDq8ikWAM",  # ElevenLabs voice ID
    api_key="your-elevenlabs-api-key"  # or set ELEVENLABS_API_KEY env var
)
response.stream_to_file(speech_file_path)
```

</TabItem>

<TabItem value="advanced" label="Advanced Features">

```python showLineNumbers title="Text-to-speech with advanced voice settings"
import litellm
from pathlib import Path

speech_file_path = Path(__file__).parent / "speech.mp3"
response = litellm.speech(
    model="elevenlabs/eleven_multilingual_v2",
    input="Hello, this is a test with custom voice settings",
    voice="21m00Tcm4TlvDq8ikWAM",
    response_format="mp3",  # Output format
    # ElevenLabs-specific voice settings
    voice_settings={
        "stability": 0.5,
        "similarity_boost": 0.8,
        "style": 0.0,
        "use_speaker_boost": True
    },
    api_key="your-elevenlabs-api-key"
)
response.stream_to_file(speech_file_path)
```

</TabItem>

<TabItem value="async" label="Async Usage">

```python showLineNumbers title="Async text-to-speech"
import litellm
import asyncio
from pathlib import Path

async def synthesize_speech():
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = await litellm.aspeech(
        model="elevenlabs/eleven_multilingual_v2",
        input="Hello, this is an async text-to-speech test",
        voice="21m00Tcm4TlvDq8ikWAM",
        api_key="your-elevenlabs-api-key"
    )
    response.stream_to_file(speech_file_path)

# Run async synthesis
asyncio.run(synthesize_speech())
```

</TabItem>
</Tabs>

### LiteLLM Proxy

#### 1. Configure your proxy

<Tabs>
<TabItem value="config-yaml" label="config.yaml">

```yaml showLineNumbers title="ElevenLabs TTS configuration in config.yaml"
model_list:
  - model_name: elevenlabs-tts
    litellm_params:
      model: elevenlabs/eleven_multilingual_v2
      api_key: os.environ/ELEVENLABS_API_KEY

general_settings:
  master_key: your-master-key
```

</TabItem>

<TabItem value="env-vars" label="Environment Variables">

```bash showLineNumbers title="Required environment variables"
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
export LITELLM_MASTER_KEY="your-master-key"
```

</TabItem>
</Tabs>

#### 2. Start the proxy

```bash showLineNumbers title="Start LiteLLM proxy server"
litellm --config config.yaml

# Proxy will be available at http://localhost:4000
```

#### 3. Make text-to-speech requests

<Tabs>
<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Text-to-speech with curl"
curl http://localhost:4000/v1/audio/speech \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "elevenlabs-tts",
    "input": "Hello, this is a test of ElevenLabs text-to-speech",
    "voice": "21m00Tcm4TlvDq8ikWAM"
  }' \
  --output speech.mp3
```

</TabItem>

<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="Using OpenAI SDK with LiteLLM proxy"
from openai import OpenAI

# Initialize client with your LiteLLM proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-api-key"
)

# Generate speech
response = client.audio.speech.create(
    model="elevenlabs-tts",
    input="Hello, this is a test of ElevenLabs text-to-speech",
    voice="21m00Tcm4TlvDq8ikWAM"
)

# Save to file
with open("speech.mp3", "wb") as f:
    f.write(response.content)
```

</TabItem>

<TabItem value="javascript" label="JavaScript/Node.js">

```javascript showLineNumbers title="Text-to-speech with JavaScript"
import OpenAI from 'openai';
import fs from 'fs';

const openai = new OpenAI({
  baseURL: 'http://localhost:4000',
  apiKey: 'your-litellm-api-key'
});

async function synthesizeSpeech() {
  const response = await openai.audio.speech.create({
    model: 'elevenlabs-tts',
    input: 'Hello, this is a test of ElevenLabs text-to-speech',
    voice: '21m00Tcm4TlvDq8ikWAM'
  });

  const buffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync('speech.mp3', buffer);
}

synthesizeSpeech();
```

</TabItem>
</Tabs>

### Supported Models

| Model | Description | Use Case |
|-------|-------------|----------|
| eleven_multilingual_v2 | High-quality TTS with 29 language support | General purpose, high quality |
| eleven_flash_v2_5 | Fast, affordable TTS with ultra-low latency | Real-time applications |
| eleven_turbo_v2_5 | Balanced quality and speed | Good balance of quality/speed |

### Voice Settings

ElevenLabs provides advanced voice customization options:

```python showLineNumbers title="Voice Settings Example"
response = litellm.speech(
    model="elevenlabs/eleven_multilingual_v2",
    input="Your text here",
    voice="21m00Tcm4TlvDq8ikWAM",
    voice_settings={
        "stability": 0.5,        # 0-1, higher = more stable
        "similarity_boost": 0.8, # 0-1, higher = more similar to original
        "style": 0.0,            # 0-1, style exaggeration
        "use_speaker_boost": True # Boost similarity to original speaker
    }
)
```


