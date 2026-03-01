# Sarvam.ai

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

LiteLLM supports all the text models from [Sarvam ai](https://docs.sarvam.ai/api-reference-docs/chat/chat-completions)

## Usage

```python
import os
from litellm import completion

# Set your Sarvam API key
os.environ["SARVAM_API_KEY"] = ""

messages = [{"role": "user", "content": "Hello"}]

response = completion(
    model="sarvam/sarvam-m",
    messages=messages,
)
print(response)
```

## Usage with LiteLLM Proxy Server

Here's how to call a Sarvam.ai model with the LiteLLM Proxy Server

1. **Modify the `config.yaml`:**

    ```yaml
    model_list:
      - model_name: my-model
        litellm_params:
          model: sarvam/<your-model-name>  # add sarvam/ prefix to route as Sarvam provider
          api_key: api-key                 # api key to send your model
    ```

2. **Start the proxy:**

    ```bash
    $ litellm --config /path/to/config.yaml
    ```

3. **Send a request to LiteLLM Proxy Server:**

    <Tabs>

    <TabItem value="openai" label="OpenAI Python v1.0.0+">

    ```python
    import openai

    client = openai.OpenAI(
        api_key="sk-1234",             # pass litellm proxy key, if you're using virtual keys
        base_url="http://0.0.0.0:4000" # litellm-proxy-base url
    )

    response = client.chat.completions.create(
        model="my-model",
        messages=[
            {
                "role": "user",
                "content": "what llm are you"
            }
        ],
    )

    print(response)
    ```
    </TabItem>

    <TabItem value="curl" label="curl">

    ```shell
    curl --location 'http://0.0.0.0:4000/chat/completions' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
        "model": "my-model",
        "messages": [
            {
            "role": "user",
            "content": "what llm are you"
            }
        ]
    }'
    ```
    </TabItem>

    </Tabs>

## Audio Transcription (STT)

Sarvam supports speech-to-text via the `/audio/transcriptions` endpoint.

### LiteLLM Python SDK

```python
from litellm import transcription
import os

os.environ["SARVAM_API_KEY"] = "your-sarvam-api-key"

with open("audio.wav", "rb") as audio_file:
    response = transcription(
        model="sarvam/saarika:v2.5",
        file=audio_file,
        language="hi-IN",  # or "unknown" for auto-detect
    )

print(response.text)
```

### LiteLLM Proxy

```yaml
model_list:
- model_name: sarvam-stt
  litellm_params:
    model: saaras:v2.5/saarika:v2.5
    api_key: os.environ/SARVAM_API_KEY
  model_info:
    mode: audio_transcription
```

```bash
curl --location 'http://0.0.0.0:4000/v1/audio/transcriptions' \
  --header 'Authorization: Bearer sk-1234' \
  --form 'file=@"audio.wav"' \
  --form 'model="saaras:v2.5"' \
  --form 'language="hi-IN"'
```

## Text-to-Speech (TTS)

Sarvam supports text-to-speech via the `/audio/speech` endpoint.

### LiteLLM Python SDK

```python
from litellm import speech
import os

os.environ["SARVAM_API_KEY"] = "your-sarvam-api-key"

audio = speech(
    model="bulbul:v3",
    input="Welcome to Sarvam AI!",
    voice="shubh",
    response_format="wav",
)

with open("output.wav", "wb") as f:
    f.write(audio.read())
```

### LiteLLM Proxy

```yaml
model_list:
- model_name: bulbul:v3
  litellm_params:
    model: bulbul:v3
    api_key: os.environ/SARVAM_API_KEY
  model_info:
    mode: audio_speech
```

```bash
curl http://0.0.0.0:4000/v1/audio/speech \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bulbul:v3",
    "input": "Welcome to Sarvam AI!",
    "voice": "anushka",
    "response_format": "wav"
  }' \
  --output output.wav
```
