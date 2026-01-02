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

GigaChat uses OAuth authentication. Set your credentials as environment variables:

```python
import os

# Required: Set credentials (base64-encoded client_id:client_secret)
os.environ['GIGACHAT_CREDENTIALS'] = "your-credentials-here"

# Optional: Set scope (default is GIGACHAT_API_PERS for personal use)
os.environ['GIGACHAT_SCOPE'] = "GIGACHAT_API_PERS"  # or GIGACHAT_API_B2B for business
```

Get your credentials at: https://developers.sber.ru/studio/

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

GigaChat supports image input via base64 or URL:

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

:::note
Available models may vary depending on your API access level (personal or business).
:::

## Limitations

- Only one function call per request (GigaChat API limitation)
- Maximum 1 image per message, 10 images total per conversation
- GigaChat API uses self-signed SSL certificates (SSL verification is disabled automatically)
