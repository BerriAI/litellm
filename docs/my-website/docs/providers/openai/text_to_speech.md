import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI - Text-to-speech

## Overview

| Feature | Supported | Notes |
|---------|-----------|-------|
| Cost Tracking | ✅ | Works with all supported models |
| Logging | ✅ | Works across all integrations |
| End-user Tracking | ✅ | |
| Fallbacks | ✅ | Works between supported models |
| Loadbalancing | ✅ | Works between supported models |
| Guardrails | ✅ | Applies to input text |
| Supported Models | tts-1, tts-1-hd, gpt-4o-mini-tts | |

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
    response = await aspeech(
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

## Supported Models 

| Model | Example |
|-------|-------------|
| tts-1 | speech(model="tts-1", voice="alloy", input="Hello, world!") |
| tts-1-hd | speech(model="tts-1-hd", voice="alloy", input="Hello, world!") |
| gpt-4o-mini-tts | speech(model="gpt-4o-mini-tts", voice="alloy", input="Hello, world!") |


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