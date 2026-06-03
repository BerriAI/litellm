import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# CometAPI

## Overview

| Property | Details |
|-------|-------|
| Description | CometAPI provides an OpenAI-compatible API for chat, embeddings, images, audio, and moderation. |
| Provider Route on LiteLLM | `cometapi/` |
| Link to Provider Doc | [CometAPI Documentation ↗](https://apidoc.cometapi.com/) |
| Base URL | `https://api.cometapi.com/v1` |
| Supported Operations | `/chat/completions`, `/embeddings`, `/image/generations`, `/audio/transcriptions`, `/audio/speech`, `/moderations` |

CometAPI also documents `/v1/audio/translations`. LiteLLM does not expose an audio translations public API or proxy route in this integration.

## Required Variables

```python showLineNumbers title="Environment Variables"
import os

os.environ["COMETAPI_KEY"] = ""  # your CometAPI API key
```

LiteLLM also accepts `COMETAPI_API_KEY` as a key alias. For a custom endpoint, set `COMETAPI_BASE_URL` or `COMETAPI_API_BASE`, or pass `api_base` directly.

## Usage - LiteLLM Python SDK

### Chat Completion

```python showLineNumbers title="CometAPI Chat Completion"
from litellm import completion

response = completion(
    model="cometapi/gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

print(response.choices[0].message.content)
```

### Streaming Chat Completion

```python showLineNumbers title="CometAPI Streaming Chat Completion"
from litellm import completion

response = completion(
    model="cometapi/gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to three."}],
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### Embeddings

```python showLineNumbers title="CometAPI Embeddings"
from litellm import embedding

response = embedding(
    model="cometapi/text-embedding-3-small",
    input=["LiteLLM routes this embedding request through CometAPI."],
)

print(len(response.data[0]["embedding"]))
print(response.usage)
```

### Image Generation

```python showLineNumbers title="CometAPI Image Generation"
from litellm import image_generation

response = image_generation(
    model="cometapi/gpt-image-1",
    prompt="A small comet over a clean API diagram",
    size="1024x1024",
    output_format="png",
    background="transparent",
)

print(response.data[0].url or response.data[0].b64_json[:64])
```

### Audio Speech

```python showLineNumbers title="CometAPI Audio Speech"
from litellm import speech

response = speech(
    model="cometapi/tts-1",
    input="LiteLLM can route speech generation through CometAPI.",
    voice="alloy",
)

with open("speech.mp3", "wb") as f:
    f.write(response.content)
```

### Audio Transcription

```python showLineNumbers title="CometAPI Audio Transcription"
from litellm import transcription

with open("speech.mp3", "rb") as audio_file:
    response = transcription(
        model="cometapi/whisper-1",
        file=audio_file,
    )

print(response.text)
```

### Moderations

```python showLineNumbers title="CometAPI Moderations"
from litellm import moderation

response = moderation(
    model="cometapi/omni-moderation-latest",
    input="I want to build a safe application.",
)

print(response.results[0].categories)
print(response.results[0].category_scores)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: comet-chat
    litellm_params:
      model: cometapi/gpt-4o-mini
      api_key: os.environ/COMETAPI_KEY
  - model_name: comet-embedding
    litellm_params:
      model: cometapi/text-embedding-3-small
      api_key: os.environ/COMETAPI_KEY
  - model_name: comet-image
    litellm_params:
      model: cometapi/gpt-image-1
      api_key: os.environ/COMETAPI_KEY
  - model_name: comet-speech
    litellm_params:
      model: cometapi/tts-1
      api_key: os.environ/COMETAPI_KEY
  - model_name: comet-transcription
    litellm_params:
      model: cometapi/whisper-1
      api_key: os.environ/COMETAPI_KEY
  - model_name: comet-moderation
    litellm_params:
      model: cometapi/omni-moderation-latest
      api_key: os.environ/COMETAPI_KEY
```

Proxy routes:

- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`
- `POST /v1/moderations`

The proxy also supports the same routes without the `/v1` prefix where LiteLLM already exposes aliases.

## Custom API Base

```python showLineNumbers title="Custom API Base"
from litellm import completion

response = completion(
    model="cometapi/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="https://api.cometapi.com/v1",
    api_key="your-cometapi-api-key",
)
```

## Official CometAPI References

- [Embeddings](https://apidoc.cometapi.com/api/text/embeddings)
- [Image generations](https://apidoc.cometapi.com/api/image/openai/images)
- [Audio speech](https://apidoc.cometapi.com/api/audio/create-speech)
- [Audio transcriptions](https://apidoc.cometapi.com/api/audio/create-transcription)
- [Moderations](https://apidoc.cometapi.com/api/content-moderation/create-moderation)
