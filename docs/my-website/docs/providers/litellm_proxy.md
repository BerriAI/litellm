import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# LiteLLM Proxy (LLM Gateway)


| Property | Details |
|-------|-------|
| Description | LiteLLM Proxy is an OpenAI-compatible gateway that allows you to interact with multiple LLM providers through a unified API. Simply use the `litellm_proxy/` prefix before the model name to route your requests through the proxy. |
| Provider Route on LiteLLM | `litellm_proxy/` (add this prefix to the model name, to route any requests to litellm_proxy - e.g. `litellm_proxy/your-model-name`) |
| Setup LiteLLM Gateway | [LiteLLM Gateway ↗](../simple_proxy) |
| Supported Endpoints |`/chat/completions`, `/completions`, `/embeddings`, `/audio/speech`, `/audio/transcriptions`, `/images`, `/images/edits`, `/rerank` |



## Required Variables

```python
os.environ["LITELLM_PROXY_API_KEY"] = "" # "sk-1234" your litellm proxy api key 
os.environ["LITELLM_PROXY_API_BASE"] = "" # "http://localhost:4000" your litellm proxy api base
```


## Usage (Non Streaming)
```python
import os 
import litellm
from litellm import completion

os.environ["LITELLM_PROXY_API_KEY"] = ""

# set custom api base to your proxy
# either set .env or litellm.api_base
# os.environ["LITELLM_PROXY_API_BASE"] = ""
litellm.api_base = "your-openai-proxy-url"


messages = [{ "content": "Hello, how are you?","role": "user"}]

# litellm proxy call
response = completion(model="litellm_proxy/your-model-name", messages)
```

## Usage - passing `api_base`, `api_key` per request

If you need to set api_base dynamically, just pass it in completions instead - completions(...,api_base="your-proxy-api-base")

```python
import os 
import litellm
from litellm import completion

os.environ["LITELLM_PROXY_API_KEY"] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]

# litellm proxy call
response = completion(
    model="litellm_proxy/your-model-name", 
    messages=messages, 
    api_base = "your-litellm-proxy-url",
    api_key = "your-litellm-proxy-api-key"
)
```
## Usage - Streaming

```python
import os 
import litellm
from litellm import completion

os.environ["LITELLM_PROXY_API_KEY"] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(
    model="litellm_proxy/your-model-name", 
    messages=messages,
    api_base = "your-litellm-proxy-url", 
    stream=True
)

for chunk in response:
    print(chunk)
```

## Embeddings

```python
import litellm

response = litellm.embedding(
    model="litellm_proxy/your-embedding-model",
    input="Hello world",
    api_base="your-litellm-proxy-url",
    api_key="your-litellm-proxy-api-key"
)
```

## Image Generation

```python
import litellm

response = litellm.image_generation(
    model="litellm_proxy/dall-e-3",
    prompt="A beautiful sunset over mountains",
    api_base="your-litellm-proxy-url",
    api_key="your-litellm-proxy-api-key"
)
```

## Image Edit

```python
import litellm

with open("your-image.png", "rb") as f:
    response = litellm.image_edit(
        model="litellm_proxy/gpt-image-1",
        prompt="Make this image a watercolor painting",
        image=[f],
        api_base="your-litellm-proxy-url",
        api_key="your-litellm-proxy-api-key",
    )
```

## Audio Transcription

```python
import litellm

response = litellm.transcription(
    model="litellm_proxy/whisper-1",
    file="your-audio-file",
    api_base="your-litellm-proxy-url",
    api_key="your-litellm-proxy-api-key"
)
```

## Text to Speech

```python
import litellm

response = litellm.speech(
    model="litellm_proxy/tts-1",
    input="Hello world",
    api_base="your-litellm-proxy-url",
    api_key="your-litellm-proxy-api-key"
)
``` 

## Rerank

```python
import litellm

import litellm

response = litellm.rerank(
    model="litellm_proxy/rerank-english-v2.0",
    query="What is machine learning?",
    documents=[
        "Machine learning is a field of study in artificial intelligence",
        "Biology is the study of living organisms"
    ],
    api_base="your-litellm-proxy-url",
    api_key="your-litellm-proxy-api-key"
)
```


## Integration with Other Libraries

LiteLLM Proxy works seamlessly with Langchain, LlamaIndex, OpenAI JS, Anthropic SDK, Instructor, and more.

[Learn how to use LiteLLM proxy with these libraries →](../proxy/user_keys)

## Send all SDK requests to LiteLLM Proxy

:::info

Requires v1.72.1 or higher.

:::

Use this when calling LiteLLM Proxy from any library / codebase already using the LiteLLM SDK.

These flags will route all requests through your LiteLLM proxy, regardless of the model specified.

When enabled, requests will use `LITELLM_PROXY_API_BASE` with `LITELLM_PROXY_API_KEY` as the authentication.

### Option 1: Set Globally in Code

```python
# Set the flag globally for all requests
litellm.use_litellm_proxy = True

response = litellm.completion(
    model="vertex_ai/gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
```

### Option 2: Control via Environment Variable

```python
# Control proxy usage through environment variable
os.environ["USE_LITELLM_PROXY"] = "True"

response = litellm.completion(
    model="vertex_ai/gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
```

### Option 3: Set Per Request

```python
# Enable proxy for specific requests only
response = litellm.completion(
    model="vertex_ai/gemini-2.0-flash-001",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    use_litellm_proxy=True
)
```

## OAuth2/JWT Authentication

If your LiteLLM Proxy requires OAuth2/JWT authentication (e.g., Azure AD, Keycloak, Okta), the SDK can automatically obtain and refresh tokens for you.

```python
import litellm
from litellm.proxy_auth import AzureADCredential, ProxyAuthHandler

litellm.proxy_auth = ProxyAuthHandler(
    credential=AzureADCredential(),
    scope="api://my-litellm-proxy/.default"
)
litellm.api_base = "https://my-proxy.example.com"

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

[Learn more about SDK Proxy Authentication (OAuth2/JWT Auto-Refresh) →](../proxy_auth)

## Sending `tags` to LiteLLM Proxy

Tags allow you to categorize and track your API requests for monitoring, debugging, and analytics purposes. You can send tags as a list of strings to the LiteLLM Proxy using the `extra_body` parameter.

### Usage

Send tags by including them in the `extra_body` parameter of your completion request:

```python showLineNumbers title="Usage"
import litellm

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    api_base="http://localhost:4000",
    api_key="sk-1234",
    extra_body={"tags": ["user:ishaan", "department:engineering", "priority:high"]}
)
```

### Async Usage

```python showLineNumbers title="Async Usage"
import litellm

response = await litellm.acompletion(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    api_base="http://localhost:4000", 
    api_key="sk-1234",
    extra_body={"tags": ["user:ishaan", "department:engineering"]}
)
```

