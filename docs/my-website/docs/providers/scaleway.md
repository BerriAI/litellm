
# Scaleway 
LiteLLM supports all [models available on Scaleway Generative APIs ↗](https://www.scaleway.com/en/docs/generative-apis/reference-content/supported-models/). 

## Usage with LiteLLM Python SDK

```python
import os
from litellm import completion 

os.environ["SCW_SECRET_KEY"] = "your-scaleway-secret-key"

messages = [{"role": "user", "content": "Write a short poem"}]
response = completion(model="scaleway/qwen3-235b-a22b-instruct-2507", messages=messages)
print(response)
```

## Usage with LiteLLM Proxy 

### 1. Set Scaleway models in config.yaml

```yaml
model_list:
  - model_name: scaleway-model
    litellm_params:
      model: scaleway/qwen3-235b-a22b-instruct-2507
      api_key: "os.environ/SCW_SECRET_KEY" # ensure you have `SCW_SECRET_KEY` in your .env
```

### 2. Start proxy 

```bash
litellm --config config.yaml
```

### 3. Query proxy 

Assuming the proxy is running on [http://localhost:4000](http://localhost:4000):
```bash
curl http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_LITELLM_MASTER_KEY" \
  -d '{
    "model": "scaleway-model",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Write a short poem"
      }
    ]
  }'
```
`-H "Authorization: Bearer YOUR_LITELLM_MASTER_KEY" ` is only required if you have set a LiteLLM master key


## Supported features

Scaleway provider supports all features in [Generative APIs reference documentation ↗](https://www.scaleway.com/en/developers/api/generative-apis/), such as streaming, structured outputs and tool calling.

## Audio transcription

Scaleway's `/audio/transcriptions` endpoint is OpenAI-compatible and works with Whisper models.

### Python SDK

```python
import os
from litellm import transcription

os.environ["SCW_SECRET_KEY"] = "your-scaleway-secret-key"

with open("speech.mp3", "rb") as audio_file:
    response = transcription(
        model="scaleway/whisper-large-v3",
        file=audio_file,
    )
print(response.text)
```

### Proxy config

```yaml
model_list:
  - model_name: scaleway-whisper
    litellm_params:
      model: scaleway/whisper-large-v3
      api_key: "os.environ/SCW_SECRET_KEY"
```

### Proxy request

```bash
curl http://localhost:4000/v1/audio/transcriptions \
  -H "Authorization: Bearer YOUR_LITELLM_MASTER_KEY" \
  -F model="scaleway-whisper" \
  -F file="@speech.mp3"
```

Supported optional params: `language`, `prompt`, `response_format`, `temperature`, `timestamp_granularities`.
