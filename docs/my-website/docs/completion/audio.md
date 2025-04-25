import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Using Audio Models

How to send / receive audio to a `/chat/completions` endpoint


## Audio Output from a model

Example for creating a human-like audio response to a prompt

<Tabs>

<TabItem label="LiteLLM Python SDK" value="Python">

```python
import os 
import base64
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"

# openai call
completion = await litellm.acompletion(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[{"role": "user", "content": "Is a golden retriever a good family dog?"}],
)

wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
with open("dog.wav", "wb") as f:
    f.write(wav_bytes)
```

</TabItem>
<TabItem label="LiteLLM Proxy Server" value="proxy">

1. Define an audio model on config.yaml

```yaml
model_list:
  - model_name: gpt-4o-audio-preview # OpenAI gpt-4o-audio-preview
    litellm_params:
      model: openai/gpt-4o-audio-preview
      api_key: os.environ/OPENAI_API_KEY 

```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Test it using the OpenAI Python SDK


```python
import base64
from openai import OpenAI

client = OpenAI(
    api_key="LITELLM_PROXY_KEY", # sk-1234
    base_url="LITELLM_PROXY_BASE" # http://0.0.0.0:4000
)

completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": "Is a golden retriever a good family dog?"
        }
    ]
)

print(completion.choices[0])

wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
with open("dog.wav", "wb") as f:
    f.write(wav_bytes)

```




</TabItem>
</Tabs>

## Audio Input to a model


<Tabs>

<TabItem label="LiteLLM Python SDK" value="Python">

```python
import base64
import requests

url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
response = requests.get(url)
response.raise_for_status()
wav_data = response.content
encoded_string = base64.b64encode(wav_data).decode("utf-8")

completion = litellm.completion(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": encoded_string, "format": "wav"},
                },
            ],
        },
    ],
)

print(completion.choices[0].message)
```

</TabItem>

<TabItem label="LiteLLM Proxy Server" value="proxy">


1. Define an audio model on config.yaml

```yaml
model_list:
  - model_name: gpt-4o-audio-preview # OpenAI gpt-4o-audio-preview
    litellm_params:
      model: openai/gpt-4o-audio-preview
      api_key: os.environ/OPENAI_API_KEY 

```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Test it using the OpenAI Python SDK


```python
import base64
from openai import OpenAI

client = OpenAI(
    api_key="LITELLM_PROXY_KEY", # sk-1234
    base_url="LITELLM_PROXY_BASE" # http://0.0.0.0:4000
)


# Fetch the audio file and convert it to a base64 encoded string
url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
response = requests.get(url)
response.raise_for_status()
wav_data = response.content
encoded_string = base64.b64encode(wav_data).decode('utf-8')

completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                { 
                    "type": "text",
                    "text": "What is in this recording?"
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "wav"
                    }
                }
            ]
        },
    ]
)

print(completion.choices[0].message)
```


</TabItem>
</Tabs>

## Checking if a model supports `audio_input` and `audio_output`

<Tabs>
<TabItem label="LiteLLM Python SDK" value="Python">

Use `litellm.supports_audio_output(model="")` -> returns `True` if model can generate audio output

Use `litellm.supports_audio_input(model="")` -> returns `True` if model can accept audio input

```python
assert litellm.supports_audio_output(model="gpt-4o-audio-preview") == True
assert litellm.supports_audio_input(model="gpt-4o-audio-preview") == True

assert litellm.supports_audio_output(model="gpt-3.5-turbo") == False
assert litellm.supports_audio_input(model="gpt-3.5-turbo") == False
```
</TabItem>

<TabItem label="LiteLLM Proxy Server" value="proxy">


1. Define vision models on config.yaml

```yaml
model_list:
  - model_name: gpt-4o-audio-preview # OpenAI gpt-4o-audio-preview
    litellm_params:
      model: openai/gpt-4o-audio-preview
      api_key: os.environ/OPENAI_API_KEY
  - model_name: llava-hf          # Custom OpenAI compatible model
    litellm_params:
      model: openai/llava-hf/llava-v1.6-vicuna-7b-hf
      api_base: http://localhost:8000
      api_key: fake-key
    model_info:
      supports_audio_output: True        # set supports_audio_output to True so /model/info returns this attribute as True
      supports_audio_input: True         # set supports_audio_input to True so /model/info returns this attribute as True
```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Call `/model_group/info` to check if your model supports `vision`

```shell
curl -X 'GET' \
  'http://localhost:4000/model_group/info' \
  -H 'accept: application/json' \
  -H 'x-api-key: sk-1234'
```

Expected Response 

```json
{
  "data": [
    {
      "model_group": "gpt-4o-audio-preview",
      "providers": ["openai"],
      "max_input_tokens": 128000,
      "max_output_tokens": 16384,
      "mode": "chat",
      "supports_audio_output": true, # 👈 supports_audio_output is true
      "supports_audio_input": true, # 👈 supports_audio_input is true
    },
    {
      "model_group": "llava-hf",
      "providers": ["openai"],
      "max_input_tokens": null,
      "max_output_tokens": null,
      "mode": null,
      "supports_audio_output": true, # 👈 supports_audio_output is true
      "supports_audio_input": true, # 👈 supports_audio_input is true
    }
  ]
}
```

</TabItem>
</Tabs>


## Response Format with Audio

Below is an example JSON data structure for a `message` you might receive from a `/chat/completions` endpoint when sending audio input to a model.

```json
{
  "index": 0,
  "message": {
    "role": "assistant",
    "content": null,
    "refusal": null,
    "audio": {
      "id": "audio_abc123",
      "expires_at": 1729018505,
      "data": "<bytes omitted>",
      "transcript": "Yes, golden retrievers are known to be ..."
    }
  },
  "finish_reason": "stop"
}
```
- `audio` If the audio output modality is requested, this object contains data about the audio response from the model
    - `audio.id` Unique identifier for the audio response
    - `audio.expires_at` The Unix timestamp (in seconds) for when this audio response will no longer be accessible on the server for use in multi-turn conversations.
    - `audio.data` Base64 encoded audio bytes generated by the model, in the format specified in the request.
    - `audio.transcript` Transcript of the audio generated by the model.
