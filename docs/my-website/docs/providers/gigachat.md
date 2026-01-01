import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# GigaChat
https://developers.sber.ru/docs/ru/gigachat/api/overview

GigaChat is Sber AI's large language model, Russia's leading LLM provider.

:::tip

**We support ALL GigaChat models, just set `model=gigachat/<any-model-on-gigachat>` as a prefix when sending litellm requests**

:::

## Supported Features

| Feature | Supported |
|---------|-----------|
| Chat Completion | Yes |
| Streaming | Yes |
| Async | Yes |
| Function Calling / Tools | Yes |
| Structured Output (JSON Schema) | Yes (via function call emulation) |
| Image Input | Yes (base64 and URL) |
| Embeddings | Yes |

## API Key

GigaChat uses OAuth authentication. You can provide credentials in several ways:

```python
# Option 1: Set credentials as environment variable
os.environ['GIGACHAT_CREDENTIALS'] = "your-credentials-here"

# Option 2: Set scope (default is GIGACHAT_API_PERS for personal use)
os.environ['GIGACHAT_SCOPE'] = "GIGACHAT_API_PERS"  # or GIGACHAT_API_B2B for business

# Option 3: Disable SSL verification if needed (for corporate networks)
os.environ['GIGACHAT_VERIFY_SSL_CERTS'] = "False"
```

### API Key Formats

LiteLLM supports several API key formats for GigaChat:

```python
# Standard credentials
api_key = "your-credentials"

# Credentials with scope
api_key = "giga-cred-your-credentials:GIGACHAT_API_B2B"

# Pre-obtained JWT token
api_key = "giga-auth-your-jwt-token"

# Username and password
api_key = "giga-user-username:password"
```

## Sample Usage

```python
from litellm import completion
import os

os.environ['GIGACHAT_CREDENTIALS'] = "your-credentials-here"

response = completion(
    model="gigachat/GigaChat-Pro",
    messages=[
       {"role": "user", "content": "Hello from LiteLLM!"}
   ],
)
print(response)
```

## Sample Usage - Streaming

```python
from litellm import completion
import os

os.environ['GIGACHAT_CREDENTIALS'] = "your-credentials-here"

response = completion(
    model="gigachat/GigaChat-Pro",
    messages=[
       {"role": "user", "content": "Hello from LiteLLM!"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Sample Usage - Function Calling

```python
from litellm import completion
import os

os.environ['GIGACHAT_CREDENTIALS'] = "your-credentials-here"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
}]

response = completion(
    model="gigachat/GigaChat-Pro",
    messages=[{"role": "user", "content": "What's the weather in Moscow?"}],
    tools=tools
)
print(response)
```

## Sample Usage - Structured Output

GigaChat supports structured output via JSON schema (emulated through function calling):

```python
from litellm import completion
import os

os.environ['GIGACHAT_CREDENTIALS'] = "your-credentials-here"

response = completion(
    model="gigachat/GigaChat-Pro",
    messages=[{"role": "user", "content": "Extract info: John is 30 years old"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        }
    }
)
print(response)  # Returns JSON: {"name": "John", "age": 30}
```

## Sample Usage - Image Input

```python
from litellm import completion
import os

os.environ['GIGACHAT_CREDENTIALS'] = "your-credentials-here"

response = completion(
    model="gigachat/GigaChat-Pro",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)
print(response)
```

## Sample Usage - Embeddings

```python
from litellm import embedding
import os

os.environ['GIGACHAT_CREDENTIALS'] = "your-credentials-here"

response = embedding(
    model="gigachat/Embeddings",
    input=["Hello world", "How are you?"]
)
print(response)
```

## Usage with LiteLLM Proxy

### 1. Set GigaChat Models on config.yaml

```yaml
model_list:
  - model_name: gigachat-pro
    litellm_params:
      model: gigachat/GigaChat-Pro
      api_key: "os.environ/GIGACHAT_CREDENTIALS"
  - model_name: gigachat-max
    litellm_params:
      model: gigachat/GigaChat-Max
      api_key: "os.environ/GIGACHAT_CREDENTIALS"
  - model_name: gigachat-embeddings
    litellm_params:
      model: gigachat/Embeddings
      api_key: "os.environ/GIGACHAT_CREDENTIALS"
```

### 2. Start Proxy

```bash
litellm --config config.yaml
```

### 3. Test it

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gigachat-pro",
    "messages": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
}'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gigachat-pro",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response)
```
</TabItem>
</Tabs>

## Supported Models

| Model Name | Description |
|------------|-------------|
| gigachat/GigaChat | Base GigaChat model |
| gigachat/GigaChat-2 | GigaChat version 2 |
| gigachat/GigaChat-Pro | Professional version with enhanced capabilities |
| gigachat/GigaChat-Max | Maximum capability model |
| gigachat/GigaChat-2-Max | Latest maximum capability model |
| gigachat/Embeddings | Text embeddings model |
| gigachat/Embeddings-2 | Embeddings version 2 |

:::note
Available models may vary depending on your API access level (personal or business).
Use `litellm.get_model_list(provider="gigachat")` to get the current list of available models.
:::

## Limitations

- Only one function call per request (GigaChat API limitation)
- Maximum 2 images per message, 10 images total per conversation
- SSL certificate verification is disabled by default (can be enabled via `GIGACHAT_VERIFY_SSL_CERTS=True`)

## Requirements

Install the GigaChat SDK:

```bash
pip install gigachat
```
