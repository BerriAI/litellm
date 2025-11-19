import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI
LiteLLM supports OpenAI Chat + Embedding calls.

:::tip
**We recommend using `litellm.responses()` / Responses API** for the latest OpenAI models (GPT-5, gpt-5-codex, o3-mini, etc.)
:::

### Required API Keys

```python
import os 
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

### Usage
```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"

# openai call
response = completion(
    model = "gpt-4o", 
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

### Usage - LiteLLM Proxy Server

Here's how to call OpenAI models with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export OPENAI_API_KEY=""
```

### 2. Start the proxy 

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo                          # The `openai/` prefix will call openai.chat.completions.create
      api_key: os.environ/OPENAI_API_KEY
  - model_name: gpt-3.5-turbo-instruct
    litellm_params:
      model: text-completion-openai/gpt-3.5-turbo-instruct # The `text-completion-openai/` prefix will call openai.completions.create
      api_key: os.environ/OPENAI_API_KEY
```
</TabItem>
<TabItem value="config-*" label="config.yaml - proxy all OpenAI models">

Use this to add all openai models with one API Key. **WARNING: This will not do any load balancing**
This means requests to `gpt-4`, `gpt-3.5-turbo` , `gpt-4-turbo-preview` will all go through this route 

```yaml
model_list:
  - model_name: "*"             # all requests where model not in your config go to this deployment
    litellm_params:
      model: openai/*           # set `openai/` to use the openai route
      api_key: os.environ/OPENAI_API_KEY
```
</TabItem>
<TabItem value="cli" label="CLI">

```bash
$ litellm --model gpt-3.5-turbo

# Server running on http://0.0.0.0:4000
```
</TabItem>

</Tabs>

### 3. Test it


<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
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
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
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
    model = "gpt-3.5-turbo",
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


### Optional Keys - OpenAI Organization, OpenAI API Base

```python
import os 
os.environ["OPENAI_ORGANIZATION"] = "your-org-id"       # OPTIONAL
os.environ["OPENAI_BASE_URL"] = "https://your_host/v1"     # OPTIONAL
```

### OpenAI Chat Completion Models

| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
| gpt-5 | `response = completion(model="gpt-5", messages=messages)` |
| gpt-5-mini | `response = completion(model="gpt-5-mini", messages=messages)` |
| gpt-5-nano | `response = completion(model="gpt-5-nano", messages=messages)` |
| gpt-5-chat | `response = completion(model="gpt-5-chat", messages=messages)` |
| gpt-5-chat-latest | `response = completion(model="gpt-5-chat-latest", messages=messages)` |
| gpt-5-2025-08-07 | `response = completion(model="gpt-5-2025-08-07", messages=messages)` |
| gpt-5-mini-2025-08-07 | `response = completion(model="gpt-5-mini-2025-08-07", messages=messages)` |
| gpt-5-nano-2025-08-07 | `response = completion(model="gpt-5-nano-2025-08-07", messages=messages)` |
| gpt-5-pro | `response = completion(model="gpt-5-pro", messages=messages)` |
| gpt-5.1 | `response = completion(model="gpt-5.1", messages=messages)` |
| gpt-5.1-codex | `response = completion(model="gpt-5.1-codex", messages=messages)` |
| gpt-5.1-codex-mini | `response = completion(model="gpt-5.1-codex-mini", messages=messages)` |
| gpt-4.1 | `response = completion(model="gpt-4.1", messages=messages)` |
| gpt-4.1-mini | `response = completion(model="gpt-4.1-mini", messages=messages)` |
| gpt-4.1-nano | `response = completion(model="gpt-4.1-nano", messages=messages)` |
| o4-mini | `response = completion(model="o4-mini", messages=messages)` |
| o3-mini | `response = completion(model="o3-mini", messages=messages)` |
| o3 | `response = completion(model="o3", messages=messages)` |
| o1-mini | `response = completion(model="o1-mini", messages=messages)` |
| o1-preview | `response = completion(model="o1-preview", messages=messages)` |
| gpt-4o-mini  | `response = completion(model="gpt-4o-mini", messages=messages)` |
| gpt-4o-mini-2024-07-18   | `response = completion(model="gpt-4o-mini-2024-07-18", messages=messages)` |
| gpt-4o   | `response = completion(model="gpt-4o", messages=messages)` |
| gpt-4o-2024-08-06   | `response = completion(model="gpt-4o-2024-08-06", messages=messages)` |
| gpt-4o-2024-05-13   | `response = completion(model="gpt-4o-2024-05-13", messages=messages)` |
| gpt-4-turbo   | `response = completion(model="gpt-4-turbo", messages=messages)` |
| gpt-4-turbo-preview   | `response = completion(model="gpt-4-0125-preview", messages=messages)` |
| gpt-4-0125-preview    | `response = completion(model="gpt-4-0125-preview", messages=messages)` |
| gpt-4-1106-preview    | `response = completion(model="gpt-4-1106-preview", messages=messages)` |
| gpt-3.5-turbo-1106    | `response = completion(model="gpt-3.5-turbo-1106", messages=messages)` |
| gpt-3.5-turbo         | `response = completion(model="gpt-3.5-turbo", messages=messages)` |
| gpt-3.5-turbo-0301    | `response = completion(model="gpt-3.5-turbo-0301", messages=messages)` |
| gpt-3.5-turbo-0613    | `response = completion(model="gpt-3.5-turbo-0613", messages=messages)` |
| gpt-3.5-turbo-16k     | `response = completion(model="gpt-3.5-turbo-16k", messages=messages)` |
| gpt-3.5-turbo-16k-0613| `response = completion(model="gpt-3.5-turbo-16k-0613", messages=messages)` |
| gpt-4                 | `response = completion(model="gpt-4", messages=messages)` |
| gpt-4-0314            | `response = completion(model="gpt-4-0314", messages=messages)` |
| gpt-4-0613            | `response = completion(model="gpt-4-0613", messages=messages)` |
| gpt-4-32k             | `response = completion(model="gpt-4-32k", messages=messages)` |
| gpt-4-32k-0314        | `response = completion(model="gpt-4-32k-0314", messages=messages)` |
| gpt-4-32k-0613        | `response = completion(model="gpt-4-32k-0613", messages=messages)` |


These also support the `OPENAI_BASE_URL` environment variable, which can be used to specify a custom API endpoint.

## OpenAI Vision Models 
| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
| gpt-4o   | `response = completion(model="gpt-4o", messages=messages)` |
| gpt-4-turbo    | `response = completion(model="gpt-4-turbo", messages=messages)` |
| gpt-4-vision-preview    | `response = completion(model="gpt-4-vision-preview", messages=messages)` |

#### Usage
```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"

# openai call
response = completion(
    model = "gpt-4-vision-preview", 
    messages=[
        {
            "role": "user",
            "content": [
                            {
                                "type": "text",
                                "text": "What’s in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                }
                            }
                        ]
        }
    ],
)

```

## PDF File Parsing

OpenAI has a new `file` message type that allows you to pass in a PDF file and have it parsed into a structured output. [Read more](https://platform.openai.com/docs/guides/pdf-files?api-mode=chat&lang=python)

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import base64
from litellm import completion

with open("draconomicon.pdf", "rb") as f:
    data = f.read()

base64_string = base64.b64encode(data).decode("utf-8")

completion = completion(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": "draconomicon.pdf",
                        "file_data": f"data:application/pdf;base64,{base64_string}",
                    }
                },
                {
                    "type": "text",
                    "text": "What is the first dragon in the book?",
                }
            ],
        },
    ],
)

print(completion.choices[0].message.content)
```

</TabItem>

<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: openai-model
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
```

2. Start the proxy

```bash
litellm --config config.yaml
```

3. Test it!

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{ 
    "model": "openai-model",
    "messages": [
        {"role": "user", "content": [
            {
                "type": "file",
                "file": {
                    "filename": "draconomicon.pdf",
                    "file_data": f"data:application/pdf;base64,{base64_string}",
                }
            }
        ]}
    ]
}'
```

</TabItem>
</Tabs>

## OpenAI Fine Tuned Models

| Model Name                | Function Call                                                          |
|---------------------------|-----------------------------------------------------------------|
| fine tuned `gpt-4-0613`    | `response = completion(model="ft:gpt-4-0613", messages=messages)`     |
| fine tuned `gpt-4o-2024-05-13` | `response = completion(model="ft:gpt-4o-2024-05-13", messages=messages)` |
| fine tuned `gpt-3.5-turbo-0125` | `response = completion(model="ft:gpt-3.5-turbo-0125", messages=messages)` |
| fine tuned `gpt-3.5-turbo-1106` | `response = completion(model="ft:gpt-3.5-turbo-1106", messages=messages)` |
| fine tuned `gpt-3.5-turbo-0613` | `response = completion(model="ft:gpt-3.5-turbo-0613", messages=messages)` |

## Getting Reasoning Content in `/chat/completions`

GPT-5 models return reasoning content when called via the Responses API. You can call these models via the `/chat/completions` endpoint by using the `openai/responses/` prefix.

<Tabs>
<TabItem value="sdk" label="SDK">
```python
response = litellm.completion(
    model="openai/responses/gpt-5-mini", # tells litellm to call the model via the Responses API
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    reasoning_effort="low",
)
```
</TabItem>

<TabItem value="proxy" label="PROXY">
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{ 
    "model": "openai/responses/gpt-5-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "reasoning_effort": "low"
}'
```
</TabItem>
</Tabs>

Expected Response:
```json
{
  "id": "chatcmpl-6382a222-43c9-40c4-856b-22e105d88075",
  "created": 1760146746,
  "model": "gpt-5-mini",
  "object": "chat.completion",
  "system_fingerprint": null,
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Paris",
        "role": "assistant",
        "tool_calls": null,
        "function_call": null,
        "reasoning_content": "**Identifying the capital**\n\nThe user wants me to think of the capital of France and write it down. That's pretty straightforward: it's Paris. There aren't any safety issues to consider here. I think it would be best to keep it concise, so maybe just \"Paris\" would suffice. I feel confident that I should just stick to that without adding anything else. So, let's write it down!",
        "provider_specific_fields": null
      }
    }
  ],
  "usage": {
    "completion_tokens": 7,
    "prompt_tokens": 18,
    "total_tokens": 25,
    "completion_tokens_details": null,
    "prompt_tokens_details": {
      "audio_tokens": null,
      "cached_tokens": 0,
      "text_tokens": null,
      "image_tokens": null
    }
  }
}

```

### Advanced: Using `reasoning_effort` with `summary` field

By default, `reasoning_effort` accepts a string value (`"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`) and only sets the effort level without including a reasoning summary.

To opt-in to the `summary` feature, you can pass `reasoning_effort` as a dictionary. **Note:** The `summary` field requires your OpenAI organization to have verification status. Using `summary` without verification will result in a 400 error from OpenAI.

<Tabs>
<TabItem value="sdk" label="SDK">
```python
# Option 1: String format (default - no summary)
response = litellm.completion(
    model="openai/responses/gpt-5-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    reasoning_effort="high"  # Only sets effort level
)

# Option 2: Dict format (with optional summary - requires org verification)
response = litellm.completion(
    model="openai/responses/gpt-5-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    reasoning_effort={"effort": "high", "summary": "auto"}  # "auto", "detailed", or "concise" (not all supported by all models)
)
```
</TabItem>

<TabItem value="proxy" label="PROXY">
```bash
# Option 1: String format (default - no summary)
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "openai/responses/gpt-5-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "reasoning_effort": "high"
}'

# Option 2: Dict format (with optional summary - requires org verification)
# summary options: "auto", "detailed", or "concise" (not all supported by all models)
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "openai/responses/gpt-5-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "reasoning_effort": {"effort": "high", "summary": "auto"}
}'
```
</TabItem>
</Tabs>

**Summary field options:**
- `"auto"`: System automatically determines the appropriate summary level based on the model
- `"concise"`: Provides a shorter summary (not supported by GPT-5 series models)
- `"detailed"`: Offers a comprehensive reasoning summary

**Note:** GPT-5 series models support `"auto"` and `"detailed"`, but do not support `"concise"`. O-series models (o3-pro, o4-mini, o3) support all three options. Some models like o3-mini and o1 do not support reasoning summaries at all.

**Supported `reasoning_effort` values by model:**

| Model | Default (when not set) | Supported Values |
|-------|----------------------|------------------|
| `gpt-5.1` | `none` | `none`, `low`, `medium`, `high` |
| `gpt-5` | `medium` | `minimal`, `low`, `medium`, `high` |
| `gpt-5-mini` | `medium` | `none`, `minimal`, `low`, `medium`, `high` |
| `gpt-5-nano` | `none` | `none`, `low`, `medium`, `high` |
| `gpt-5-codex` | `adaptive` | `low`, `medium`, `high` (no `minimal`) |
| `gpt-5.1-codex` | `adaptive` | `low`, `medium`, `high` (no `minimal`) |
| `gpt-5.1-codex-mini` | `adaptive` | `low`, `medium`, `high` (no `minimal`) |
| `gpt-5-pro` | `high` | `high` only |

**Note:**
- GPT-5.1 introduced a new `reasoning_effort="none"` setting for faster, lower-latency responses. This replaces the `"minimal"` setting from GPT-5.
- `gpt-5-pro` only accepts `reasoning_effort="high"`. Other values will return an error.
- When `reasoning_effort` is not set (None), OpenAI defaults to the value shown in the "Default" column.

See [OpenAI Reasoning documentation](https://platform.openai.com/docs/guides/reasoning) for more details on organization verification requirements.

### Verbosity Control for GPT-5 Models

The `verbosity` parameter controls the length and detail of responses from GPT-5 family models. It accepts three values: `"low"`, `"medium"`, or `"high"`.

**Supported models:** `gpt-5`, `gpt-5.1`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-pro`

**Note:** GPT-5-Codex models (`gpt-5-codex`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`) do **not** support the `verbosity` parameter.

**Use cases:**
- **`"low"`**: Best for concise answers or simple code generation (e.g., SQL queries)
- **`"medium"`**: Default - balanced output length
- **`"high"`**: Use when you need thorough explanations or extensive code refactoring

<Tabs>
<TabItem value="sdk" label="SDK">
```python
import litellm

# Low verbosity - concise responses
response = litellm.completion(
    model="gpt-5.1",
    messages=[{"role": "user", "content": "Write a function to reverse a string"}],
    verbosity="low"
)

# High verbosity - detailed responses
response = litellm.completion(
    model="gpt-5.1",
    messages=[{"role": "user", "content": "Explain how neural networks work"}],
    verbosity="high"
)
```
</TabItem>

<TabItem value="proxy" label="PROXY">
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-5.1",
    "messages": [{"role": "user", "content": "Write a function to reverse a string"}],
    "verbosity": "low"
}'
```
</TabItem>
</Tabs>


## OpenAI Chat Completion to Responses API Bridge

Call any Responses API model from OpenAI's `/chat/completions` endpoint. 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
import os 

os.environ["OPENAI_API_KEY"] = "sk-1234"

response = litellm.completion(
    model="o3-deep-research-2025-06-26",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    tools=[
        {"type": "web_search_preview"},
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ],
)
print(response)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: openai-model
    litellm_params:
      model: o3-deep-research-2025-06-26
      api_key: os.environ/OPENAI_API_KEY
```

2. Start the proxy

```bash
litellm --config config.yaml
```

3. Test it!

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{ 
    "model": "openai-model",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "tools": [
        {"type": "web_search_preview"},
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ],
}'
```

</TabItem>
</Tabs>


## OpenAI Audio Transcription

LiteLLM supports OpenAI Audio Transcription endpoint.

Supported models:

| Model Name                | Function Call                                                          |
|---------------------------|-----------------------------------------------------------------|
| `whisper-1`    | `response = completion(model="whisper-1", file=audio_file)`     |
| `gpt-4o-transcribe` | `response = completion(model="gpt-4o-transcribe", file=audio_file)` |
| `gpt-4o-mini-transcribe` | `response = completion(model="gpt-4o-mini-transcribe", file=audio_file)` |

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import transcription
import os 

# set api keys 
os.environ["OPENAI_API_KEY"] = ""
audio_file = open("/path/to/audio.mp3", "rb")

response = transcription(model="gpt-4o-transcribe", file=audio_file)

print(f"response: {response}")
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
- model_name: gpt-4o-transcribe
  litellm_params:
    model: gpt-4o-transcribe
    api_key: os.environ/OPENAI_API_KEY
  model_info:
    mode: audio_transcription
    
general_settings:
  master_key: sk-1234
```

2. Start the proxy

```bash
litellm --config config.yaml
```

3. Test it!

```bash
curl --location 'http://0.0.0.0:8000/v1/audio/transcriptions' \
--header 'Authorization: Bearer sk-1234' \
--form 'file=@"/Users/krrishdholakia/Downloads/gettysburg.wav"' \
--form 'model="gpt-4o-transcribe"'
```



</TabItem>
</Tabs>



## Advanced

### Getting OpenAI API Response Headers 

Set `litellm.return_response_headers = True` to get raw response headers from OpenAI

You can expect to always get the `_response_headers` field from `litellm.completion()`, `litellm.embedding()` functions

<Tabs>
<TabItem value="litellm.completion" label="litellm.completion">

```python
litellm.return_response_headers = True

# /chat/completion
response = completion(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "hi",
        }
    ],
)
print(f"response: {response}")
print("_response_headers=", response._response_headers)
```
</TabItem>

<TabItem value="litellm.completion - streaming" label="litellm.completion + stream">

```python
litellm.return_response_headers = True

# /chat/completion
response = completion(
    model="gpt-4o-mini",
    stream=True,
    messages=[
        {
            "role": "user",
            "content": "hi",
        }
    ],
)
print(f"response: {response}")
print("response_headers=", response._response_headers)
for chunk in response:
    print(chunk)
```
</TabItem>

<TabItem value="litellm.embedding" label="litellm.embedding">

```python
litellm.return_response_headers = True

# embedding
embedding_response = litellm.embedding(
    model="text-embedding-ada-002",
    input="hello",
)

embedding_response_headers = embedding_response._response_headers
print("embedding_response_headers=", embedding_response_headers)
```

</TabItem>
</Tabs>
Expected Response Headers from OpenAI

```json
{
  "date": "Sat, 20 Jul 2024 22:05:23 GMT",
  "content-type": "application/json",
  "transfer-encoding": "chunked",
  "connection": "keep-alive",
  "access-control-allow-origin": "*",
  "openai-model": "text-embedding-ada-002",
  "openai-organization": "*****",
  "openai-processing-ms": "20",
  "openai-version": "2020-10-01",
  "strict-transport-security": "max-age=15552000; includeSubDomains; preload",
  "x-ratelimit-limit-requests": "5000",
  "x-ratelimit-limit-tokens": "5000000",
  "x-ratelimit-remaining-requests": "4999",
  "x-ratelimit-remaining-tokens": "4999999",
  "x-ratelimit-reset-requests": "12ms",
  "x-ratelimit-reset-tokens": "0s",
  "x-request-id": "req_cc37487bfd336358231a17034bcfb4d9",
  "cf-cache-status": "DYNAMIC",
  "set-cookie": "__cf_bm=E_FJY8fdAIMBzBE2RZI2.OkMIO3lf8Hz.ydBQJ9m3q8-1721513123-1.0.1.1-6OK0zXvtd5s9Jgqfz66cU9gzQYpcuh_RLaUZ9dOgxR9Qeq4oJlu.04C09hOTCFn7Hg.k.2tiKLOX24szUE2shw; path=/; expires=Sat, 20-Jul-24 22:35:23 GMT; domain=.api.openai.com; HttpOnly; Secure; SameSite=None, *cfuvid=SDndIImxiO3U0aBcVtoy1TBQqYeQtVDo1L6*Nlpp7EU-1721513123215-0.0.1.1-604800000; path=/; domain=.api.openai.com; HttpOnly; Secure; SameSite=None",
  "x-content-type-options": "nosniff",
  "server": "cloudflare",
  "cf-ray": "8a66409b4f8acee9-SJC",
  "content-encoding": "br",
  "alt-svc": "h3=\":443\"; ma=86400"
}
```

### Parallel Function calling
See a detailed walthrough of parallel function calling with litellm [here](https://docs.litellm.ai/docs/completion/function_call)
```python
import litellm
import json
# set openai api key
import os
os.environ['OPENAI_API_KEY'] = "" # litellm reads OPENAI_API_KEY from .env and sends the request
# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = litellm.completion(
    model="gpt-3.5-turbo-1106",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
print("\nLLM Response1:\n", response)
response_message = response.choices[0].message
tool_calls = response.choices[0].message.tool_calls
```

### Setting `extra_headers` for completion calls
```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"

response = completion(
    model = "gpt-3.5-turbo", 
    messages=[{ "content": "Hello, how are you?","role": "user"}],
    extra_headers={"AI-Resource Group": "ishaan-resource"}
)
```

### Setting Organization-ID for completion calls
This can be set in one of the following ways:
- Environment Variable `OPENAI_ORGANIZATION`
- Params to `litellm.completion(model=model, organization="your-organization-id")`
- Set as `litellm.organization="your-organization-id"`
```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_ORGANIZATION"] = "your-org-id" # OPTIONAL

response = completion(
    model = "gpt-3.5-turbo", 
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

### Set `ssl_verify=False`

This is done by setting your own `httpx.Client` 

- For `litellm.completion` set `litellm.client_session=httpx.Client(verify=False)`
- For `litellm.acompletion` set `litellm.aclient_session=AsyncClient.Client(verify=False)`
```python
import litellm, httpx

# for completion
litellm.client_session = httpx.Client(verify=False)
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=messages,
)

# for acompletion
litellm.aclient_session = httpx.AsyncClient(verify=False)
response = litellm.acompletion(
    model="gpt-3.5-turbo",
    messages=messages,
)
```


### Using OpenAI Proxy with LiteLLM
```python
import os 
import litellm
from litellm import completion

os.environ["OPENAI_API_KEY"] = ""

# set custom api base to your proxy
# either set .env or litellm.api_base
# os.environ["OPENAI_BASE_URL"] = "https://your_host/v1"
litellm.api_base = "https://your_host/v1"


messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion("openai/your-model-name", messages)
```

If you need to set api_base dynamically, just pass it in completions instead - `completions(...,api_base="your-proxy-api-base")`

For more check out [setting API Base/Keys](../set_keys.md)

### Forwarding Org ID for Proxy requests

Forward openai Org ID's from the client to OpenAI with `forward_openai_org_id` param. 

1. Setup config.yaml 

```yaml
model_list:
  - model_name: "gpt-3.5-turbo"
    litellm_params:
      model: gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

general_settings:
    forward_openai_org_id: true # 👈 KEY CHANGE
```

2. Start Proxy

```bash
litellm --config config.yaml --detailed_debug

# RUNNING on http://0.0.0.0:4000
```

3. Make OpenAI call

```python
from openai import OpenAI
client = OpenAI(
    api_key="sk-1234",
    organization="my-special-org",
    base_url="http://0.0.0.0:4000"
)

client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])
```

In your logs you should see the forwarded org id

```bash
LiteLLM:DEBUG: utils.py:255 - Request to litellm:
LiteLLM:DEBUG: utils.py:255 - litellm.acompletion(... organization='my-special-org',)
```

## GPT-5 Pro Special Notes

GPT-5 Pro is OpenAI's most advanced reasoning model with unique characteristics:

- **Responses API Only**: GPT-5 Pro is only available through the `/v1/responses` endpoint
- **No Streaming**: Does not support streaming responses
- **High Reasoning**: Designed for complex reasoning tasks with highest effort reasoning
- **Context Window**: 400,000 tokens input, 272,000 tokens output
- **Pricing**: $15.00 input / $120.00 output per 1M tokens (Standard), $7.50 input / $60.00 output (Batch)
- **Tools**: Supports Web Search, File Search, Image Generation, MCP (but not Code Interpreter or Computer Use)
- **Modalities**: Text and Image input, Text output only

```python
# GPT-5 Pro usage example
response = completion(
    model="gpt-5-pro", 
    messages=[{"role": "user", "content": "Solve this complex reasoning problem..."}]
)
```

## Video Generation

LiteLLM supports OpenAI's video generation models including Sora.

For detailed documentation on video generation, see [OpenAI Video Generation →](./openai/video_generation.md)