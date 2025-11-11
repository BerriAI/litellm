import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Fireworks AI


:::info
**We support ALL Fireworks AI models, just set `fireworks_ai/` as a prefix when sending completion requests**
:::

| Property | Details |
|-------|-------|
| Description | The fastest and most efficient inference engine to build production-ready, compound AI systems. |
| Provider Route on LiteLLM | `fireworks_ai/` |
| Provider Doc | [Fireworks AI ↗](https://docs.fireworks.ai/getting-started/introduction) |
| Supported OpenAI Endpoints | `/chat/completions`, `/embeddings`, `/completions`, `/audio/transcriptions` |


## Overview

This guide explains how to integrate LiteLLM with Fireworks AI. You can connect to Fireworks AI in three main ways:

1. <b> Using Fireworks AI serverless models </b> – Easy connection to Fireworks-managed models.
2. <b> Connecting to a model in your own Fireworks account </b> – Access models that are hosted within your Fireworks account.
3. <b> Connecting via a direct-route deployment </b> – A more flexible, customizable connection to a specific Fireworks instance.


## API Key
```python
# env variable
os.environ['FIREWORKS_AI_API_KEY']
```

## Sample Usage - Serverless Models
```python
from litellm import completion
import os

os.environ['FIREWORKS_AI_API_KEY'] = ""
response = completion(
    model="fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Serverless Models - Streaming
```python
from litellm import completion
import os

os.environ['FIREWORKS_AI_API_KEY'] = ""
response = completion(
    model="fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Sample Usage -  Models in Your Own Fireworks Account 
```python
from litellm import completion
import os

os.environ['FIREWORKS_AI_API_KEY'] = ""
response = completion(
    model="fireworks_ai/accounts/fireworks/models/YOUR_MODEL_ID", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Direct-Route Deployment
```python
from litellm import completion
import os

os.environ['FIREWORKS_AI_API_KEY'] = "YOUR_DIRECT_API_KEY"
response = completion(
    model="fireworks_ai/accounts/fireworks/models/qwen2p5-coder-7b#accounts/gitlab/deployments/2fb7764c", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
   api_base="https://gitlab-2fb7764c.direct.fireworks.ai/v1"
)
print(response)
```

> **Note:** The above is for the chat interface, if you want to use the text completion interface it's model="text-completion-openai/accounts/fireworks/models/qwen2p5-coder-7b#accounts/gitlab/deployments/2fb7764c"


## Usage with LiteLLM Proxy 

### 1. Set Fireworks AI Models on config.yaml

```yaml
model_list:
  - model_name: fireworks-llama-v3-70b-instruct
    litellm_params:
      model: fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct
      api_key: "os.environ/FIREWORKS_AI_API_KEY"
```

### 2. Start Proxy 

```
litellm --config config.yaml
```

### 3. Test it


<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "fireworks-llama-v3-70b-instruct",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="fireworks-llama-v3-70b-instruct", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)

```
</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000", # set openai_api_base to the LiteLLM Proxy
    model = "fireworks-llama-v3-70b-instruct",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```
</TabItem>
</Tabs>

## Document Inlining 

LiteLLM supports document inlining for Fireworks AI models. This is useful for models that are not vision models, but still need to parse documents/images/etc.

LiteLLM will add `#transform=inline` to the url of the image_url, if the model is not a vision model.[**See Code**](https://github.com/BerriAI/litellm/blob/1ae9d45798bdaf8450f2dfdec703369f3d2212b7/litellm/llms/fireworks_ai/chat/transformation.py#L114)

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ["FIREWORKS_AI_API_KEY"] = "YOUR_API_KEY"
os.environ["FIREWORKS_AI_API_BASE"] = "https://audio-prod.api.fireworks.ai/v1"

completion = litellm.completion(
    model="fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://storage.googleapis.com/fireworks-public/test/sample_resume.pdf"
                    },
                },
                {
                    "type": "text",
                    "text": "What are the candidate's BA and MBA GPAs?",
                },
            ],
        }
    ],
)
print(completion)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: llama-v3p3-70b-instruct
    litellm_params:
      model: fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct
      api_key: os.environ/FIREWORKS_AI_API_KEY
    #   api_base: os.environ/FIREWORKS_AI_API_BASE [OPTIONAL], defaults to "https://api.fireworks.ai/inference/v1"
```

2. Start Proxy

```
litellm --config config.yaml
```

3. Test it

```bash
curl -L -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer YOUR_API_KEY' \
-d '{"model": "llama-v3p3-70b-instruct", 
    "messages": [        
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://storage.googleapis.com/fireworks-public/test/sample_resume.pdf"
                    },
                },
                {
                    "type": "text",
                    "text": "What are the candidate's BA and MBA GPAs?",
                },
            ],
        }
    ]}'
```

</TabItem>
</Tabs>

### Disable Auto-add

If you want to disable the auto-add of `#transform=inline` to the url of the image_url, you can set the `auto_add_transform_inline` to `False` in the `FireworksAIConfig` class.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
litellm.disable_add_transform_inline_image_block = True
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
litellm_settings:
    disable_add_transform_inline_image_block: true
```

</TabItem>
</Tabs>

## Supported Models - ALL Fireworks AI Models Supported!

:::info
We support ALL Fireworks AI models, just set `fireworks_ai/` as a prefix when sending completion requests
:::

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| llama-v3p2-1b-instruct | `completion(model="fireworks_ai/llama-v3p2-1b-instruct", messages)` |
| llama-v3p2-3b-instruct | `completion(model="fireworks_ai/llama-v3p2-3b-instruct", messages)` |
| llama-v3p2-11b-vision-instruct | `completion(model="fireworks_ai/llama-v3p2-11b-vision-instruct", messages)` |
| llama-v3p2-90b-vision-instruct | `completion(model="fireworks_ai/llama-v3p2-90b-vision-instruct", messages)` |
| mixtral-8x7b-instruct | `completion(model="fireworks_ai/mixtral-8x7b-instruct", messages)` | 
| firefunction-v1 | `completion(model="fireworks_ai/firefunction-v1", messages)` |
| llama-v2-70b-chat | `completion(model="fireworks_ai/llama-v2-70b-chat", messages)` |  

## Supported Embedding Models

:::info
We support ALL Fireworks AI models, just set `fireworks_ai/` as a prefix when sending embedding requests
:::

| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
| fireworks_ai/nomic-ai/nomic-embed-text-v1.5 | `response = litellm.embedding(model="fireworks_ai/nomic-ai/nomic-embed-text-v1.5", input=input_text)` |
| fireworks_ai/nomic-ai/nomic-embed-text-v1 | `response = litellm.embedding(model="fireworks_ai/nomic-ai/nomic-embed-text-v1", input=input_text)` |
| fireworks_ai/WhereIsAI/UAE-Large-V1 | `response = litellm.embedding(model="fireworks_ai/WhereIsAI/UAE-Large-V1", input=input_text)` |
| fireworks_ai/thenlper/gte-large | `response = litellm.embedding(model="fireworks_ai/thenlper/gte-large", input=input_text)` |
| fireworks_ai/thenlper/gte-base | `response = litellm.embedding(model="fireworks_ai/thenlper/gte-base", input=input_text)` |


## Audio Transcription

### Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import transcription
import os

os.environ["FIREWORKS_AI_API_KEY"] = "YOUR_API_KEY"
os.environ["FIREWORKS_AI_API_BASE"] = "https://audio-prod.api.fireworks.ai/v1"

response = transcription(
    model="fireworks_ai/whisper-v3",
    audio=audio_file,
)
```

[Pass API Key/API Base in `.transcription`](../set_keys.md#passing-args-to-completion)

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: whisper-v3
    litellm_params:
      model: fireworks_ai/whisper-v3
      api_base: https://audio-prod.api.fireworks.ai/v1
      api_key: os.environ/FIREWORKS_API_KEY
    model_info:
      mode: audio_transcription
```

2. Start Proxy

```
litellm --config config.yaml
```

3. Test it

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/audio/transcriptions' \
-H 'Authorization: Bearer sk-1234' \
-F 'file=@"/Users/krrishdholakia/Downloads/gettysburg.wav"' \
-F 'model="whisper-v3"' \
-F 'response_format="verbose_json"' \
```

</TabItem>
</Tabs>