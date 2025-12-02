import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vertex AI Text to Speech (Gemini TTS, Chirp3)

## **Gemini TTS (Text-to-Speech) Audio Output**

:::info

LiteLLM supports Gemini TTS models on Vertex AI that can generate audio responses using the OpenAI-compatible `audio` parameter format.

:::

### Supported Models

LiteLLM supports Gemini TTS models with audio capabilities on Vertex AI (e.g. `vertex_ai/gemini-2.5-flash-preview-tts` and `vertex_ai/gemini-2.5-pro-preview-tts`). For the complete list of available TTS models and voices, see the [official Gemini TTS documentation](https://ai.google.dev/gemini-api/docs/speech-generation).

### Limitations

:::warning

**Important Limitations**:
- Gemini TTS models only support the `pcm16` audio format
- **Streaming support has not been added** to TTS models yet
- The `modalities` parameter must be set to `['audio']` for TTS requests

:::

### Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import json

## GET CREDENTIALS
file_path = 'path/to/vertex_ai_service_account.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)

response = completion(
    model="vertex_ai/gemini-2.5-flash-preview-tts",
    messages=[{"role": "user", "content": "Say hello in a friendly voice"}],
    modalities=["audio"],  # Required for TTS models
    audio={
        "voice": "Kore",
        "format": "pcm16"  # Required: must be "pcm16"
    },
    vertex_credentials=vertex_credentials_json
)

print(response)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: gemini-tts-flash
    litellm_params:
      model: vertex_ai/gemini-2.5-flash-preview-tts
      vertex_project: "your-project-id"
      vertex_location: "us-central1"
      vertex_credentials: "/path/to/service_account.json"
  - model_name: gemini-tts-pro
    litellm_params:
      model: vertex_ai/gemini-2.5-pro-preview-tts
      vertex_project: "your-project-id"
      vertex_location: "us-central1"
      vertex_credentials: "/path/to/service_account.json"
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Make TTS request

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "gemini-tts-flash",
    "messages": [{"role": "user", "content": "Say hello in a friendly voice"}],
    "modalities": ["audio"],
    "audio": {
      "voice": "Kore",
      "format": "pcm16"
    }
  }'
```

</TabItem>
</Tabs>

### Advanced Usage

You can combine TTS with other Gemini features:

```python
response = completion(
    model="vertex_ai/gemini-2.5-pro-preview-tts",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that speaks clearly."},
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    modalities=["audio"],
    audio={
        "voice": "Charon",
        "format": "pcm16"
    },
    temperature=0.7,
    max_tokens=150,
    vertex_credentials=vertex_credentials_json
)
```

For more information about Gemini's TTS capabilities and available voices, see the [official Gemini TTS documentation](https://ai.google.dev/gemini-api/docs/speech-generation).


## **Text to Speech API (Chirp3)**

:::info

LiteLLM supports calling [Vertex AI Text to Speech API](https://console.cloud.google.com/vertex-ai/generative/speech/text-to-speech) in the OpenAI text to speech API format

:::


### Usage - Basic

<Tabs>
<TabItem value="sdk" label="SDK">

Vertex AI does not support passing a `model` param - so passing `model=vertex_ai/` is the only required param

**Sync Usage**

```python
speech_file_path = Path(__file__).parent / "speech_vertex.mp3"
response = litellm.speech(
    model="vertex_ai/",
    input="hello what llm guardrail do you have",
)
response.stream_to_file(speech_file_path)
```

**Async Usage**
```python
speech_file_path = Path(__file__).parent / "speech_vertex.mp3"
response = litellm.aspeech(
    model="vertex_ai/",
    input="hello what llm guardrail do you have",
)
response.stream_to_file(speech_file_path)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY (Unified Endpoint)">

1. Add model to config.yaml
```yaml
model_list:
  - model_name: vertex-tts
    litellm_params:
      model: vertex_ai/ # Vertex AI does not support passing a `model` param - so passing `model=vertex_ai/` is the only required param
      vertex_project: "adroit-crow-413218"
      vertex_location: "us-central1"
      vertex_credentials: adroit-crow-413218-a956eef1a2a8.json 

litellm_settings:
  drop_params: True
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request use OpenAI Python SDK


```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# see supported values for "voice" on vertex here: 
# https://console.cloud.google.com/vertex-ai/generative/speech/text-to-speech
response = client.audio.speech.create(
    model = "vertex-tts",
    input="the quick brown fox jumped over the lazy dogs",
    voice={'languageCode': 'en-US', 'name': 'en-US-Studio-O'}
)
print("response from proxy", response)
```

</TabItem>
</Tabs>


### Usage - `ssml` as input

Pass your `ssml` as input to the `input` param, if it contains `<speak>`, it will be automatically detected and passed as `ssml` to the Vertex AI API

If you need to force your `input` to be passed as `ssml`, set `use_ssml=True`

<Tabs>
<TabItem value="sdk" label="SDK">

Vertex AI does not support passing a `model` param - so passing `model=vertex_ai/` is the only required param


```python
speech_file_path = Path(__file__).parent / "speech_vertex.mp3"


ssml = """
<speak>
    <p>Hello, world!</p>
    <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
</speak>
"""

response = litellm.speech(
    input=ssml,
    model="vertex_ai/test",
    voice={
        "languageCode": "en-UK",
        "name": "en-UK-Studio-O",
    },
    audioConfig={
        "audioEncoding": "LINEAR22",
        "speakingRate": "10",
    },
)
response.stream_to_file(speech_file_path)
```

</TabItem>

<TabItem value="proxy" label="LiteLLM PROXY (Unified Endpoint)">

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

ssml = """
<speak>
    <p>Hello, world!</p>
    <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
</speak>
"""

# see supported values for "voice" on vertex here: 
# https://console.cloud.google.com/vertex-ai/generative/speech/text-to-speech
response = client.audio.speech.create(
    model = "vertex-tts",
    input=ssml,
    voice={'languageCode': 'en-US', 'name': 'en-US-Studio-O'},
)
print("response from proxy", response)
```

</TabItem>
</Tabs>


### Forcing SSML Usage

You can force the use of SSML by setting the `use_ssml` parameter to `True`. This is useful when you want to ensure that your input is treated as SSML, even if it doesn't contain the `<speak>` tags.

Here are examples of how to force SSML usage:


<Tabs>
<TabItem value="sdk" label="SDK">

Vertex AI does not support passing a `model` param - so passing `model=vertex_ai/` is the only required param


```python
speech_file_path = Path(__file__).parent / "speech_vertex.mp3"


ssml = """
<speak>
    <p>Hello, world!</p>
    <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
</speak>
"""

response = litellm.speech(
    input=ssml,
    use_ssml=True,
    model="vertex_ai/test",
    voice={
        "languageCode": "en-UK",
        "name": "en-UK-Studio-O",
    },
    audioConfig={
        "audioEncoding": "LINEAR22",
        "speakingRate": "10",
    },
)
response.stream_to_file(speech_file_path)
```

</TabItem>

<TabItem value="proxy" label="LiteLLM PROXY (Unified Endpoint)">

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

ssml = """
<speak>
    <p>Hello, world!</p>
    <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
</speak>
"""

# see supported values for "voice" on vertex here: 
# https://console.cloud.google.com/vertex-ai/generative/speech/text-to-speech
response = client.audio.speech.create(
    model = "vertex-tts",
    input=ssml, # pass as None since OpenAI SDK requires this param
    voice={'languageCode': 'en-US', 'name': 'en-US-Studio-O'},
    extra_body={"use_ssml": True},
)
print("response from proxy", response)
```

</TabItem>
</Tabs>
)