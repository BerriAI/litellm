import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ElevenLabs

ElevenLabs provides high-quality AI voice technology, including speech-to-text and text-to-speech capabilities through their comprehensive audio APIs.

| Property | Details |
|----------|---------|
| Description | ElevenLabs offers advanced AI voice technology with speech-to-text transcription capabilities and text-to-speech synthesis supporting multiple languages, speaker diarization, voice cloning, and custom voice creation. |
| Provider Route on LiteLLM | `elevenlabs/` |
| Provider Doc | [ElevenLabs API ↗](https://elevenlabs.io/docs/api-reference) |
| Supported Endpoints | `/audio/transcriptions`, `/audio/speech` |

## Quick Start - Speech-to-Text

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

### Response Format

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

## Quick Start - Text-to-Speech

### LiteLLM Python SDK

<Tabs>
<TabItem value="basic-usage" label="Basic Usage">

```python showLineNumbers title="Basic text-to-speech with ElevenLabs"
import litellm

# Generate speech from text
response = litellm.speech(
    model="elevenlabs/eleven_multilingual_v2",
    voice="Xb7hH8MSUJpSbSDYk0k2",  # Voice ID from ElevenLabs
    input="Hello, this is a test of ElevenLabs text-to-speech synthesis.",
    api_key="your-elevenlabs-api-key"  # or set ELEVENLABS_API_KEY env var
)

# Save audio to file
response.write_to_file("output.mp3")
```

</TabItem>

<TabItem value="advanced-features" label="Advanced Features">

```python showLineNumbers title="Text-to-speech with advanced options"
import litellm

# Generate speech with custom parameters
response = litellm.speech(
    model="elevenlabs/eleven_multilingual_v2",
    voice="Xb7hH8MSUJpSbSDYk0k2",
    input="Welcome to advanced ElevenLabs TTS with custom settings.",
    output_format="mp3_44100_128",  # Specify audio format
    speed=1.2,                      # Speech speed: 1.0 = normal, <1.0 = slower, >1.0 = faster
    api_key="your-elevenlabs-api-key"
)

response.write_to_file("advanced_output.mp3")
```

</TabItem>

<TabItem value="async-usage" label="Async Usage">

```python showLineNumbers title="Async text-to-speech"
import litellm
import asyncio

async def generate_speech():
    response = await litellm.aspeech(
        model="elevenlabs/eleven_multilingual_v2", 
        voice="Xb7hH8MSUJpSbSDYk0k2",
        input="This is async text-to-speech generation.",
        api_key="your-elevenlabs-api-key"
    )
    
    response.write_to_file("async_output.mp3")
    return "Speech generated successfully!"

# Run async speech generation
result = asyncio.run(generate_speech())
print(result)
```

</TabItem>
</Tabs>

#### Supported Audio Formats

ElevenLabs supports a wide range of audio formats. For the complete list of available formats, see the [ElevenLabs Text-to-Speech API documentation](https://elevenlabs.io/docs/api-reference/text-to-speech/convert).

**OpenAI Compatibility**: LiteLLM automatically maps OpenAI-style `response_format` parameters to ElevenLabs formats:

- **`mp3`** → `mp3_44100_128` (Default format)
- **`pcm`** → `pcm_44100` (Uncompressed audio)  
- **`opus`** → `opus_48000_128` (Web-optimized format)

**ElevenLabs Native Formats**: You can also specify ElevenLabs formats directly using `output_format`:

```python
response = litellm.speech(
    model="elevenlabs/eleven_multilingual_v2",
    voice="Xb7hH8MSUJpSbSDYk0k2",
    input="Hello, world!",
    output_format="mp3_44100_192",  # High-quality MP3
    api_key="your-elevenlabs-api-key"
)
```

Supported ElevenLabs formats include: `mp3_44100_128`, `mp3_44100_192`, `pcm_8000`, `pcm_44100`, `ulaw_8000`, `alaw_8000`, `opus_48000_128`, and more.

#### Model Selection

Choose from ElevenLabs' available TTS models by checking the [ElevenLabs Models documentation](https://elevenlabs.io/docs/models):
- Browse available models and their capabilities
- Copy the Model ID for your chosen model
- Use the ID with the `elevenlabs/` prefix (e.g., `elevenlabs/eleven_multilingual_v2`)

Popular models include:
- `eleven_multilingual_v2` - Most lifelike model with rich emotional expression (28+ languages)
- `eleven_turbo_v2_5` - High quality, low-latency model with good balance of quality and speed (~250ms-300ms)
- `eleven_flash_v2_5` - Ultra-fast model optimized for real-time use (~75ms)

#### Voice IDs

Get voice IDs from [ElevenLabs Default Voices](https://elevenlabs.io/app/default-voices):
- Browse available voices in the ElevenLabs dashboard
- Copy the Voice ID from your selected voice
- Use the ID in the `voice` parameter

**Voice Management**: Learn how to create, customize, and manage voices with ElevenLabs in their [Voice Management documentation](https://elevenlabs.io/docs/capabilities/voices). This includes:
- Creating custom voices
- Voice cloning capabilities
- Managing voice libraries
- Customizing voice settings

### LiteLLM Proxy

#### 1. Configure your proxy

<Tabs>
<TabItem value="config-yaml" label="config.yaml">

```yaml showLineNumbers title="ElevenLabs configuration in config.yaml"
model_list:
  - model_name: elevenlabs-tts
    litellm_params:
      model: elevenlabs/eleven_multilingual_v2
      api_key: your-elevenlabs-api-key

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

```bash showLineNumbers title="Text-to-speech with curl"
curl http://localhost:4000/v1/audio/speech \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "elevenlabs-tts",
    "voice": "Xb7hH8MSUJpSbSDYk0k2",
    "input": "Hello, this is ElevenLabs TTS via LiteLLM proxy.",
    "response_format": "mp3"
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

# Generate speech from text
response = client.audio.speech.create(
    model="elevenlabs-tts",
    voice="Xb7hH8MSUJpSbSDYk0k2",
    input="Hello, this is ElevenLabs TTS via LiteLLM proxy.",
    response_format="mp3",
)
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

async function generateSpeech() {
  const response = await openai.audio.speech.create({
    model: 'elevenlabs-tts',
    voice: 'Xb7hH8MSUJpSbSDYk0k2',
    input: 'Hello, this is ElevenLabs TTS via LiteLLM proxy.',
    response_format: 'mp3'
  });

  // Save audio to file
  const buffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync('speech.mp3', buffer);
  console.log('Speech generated and saved to speech.mp3');
}

generateSpeech();
```

</TabItem>
</Tabs>

## Troubleshooting

### Common Issues

1. **Invalid API Key**: Ensure `ELEVENLABS_API_KEY` is set correctly


