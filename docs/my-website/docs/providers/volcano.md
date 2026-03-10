# Volcano Engine (Volcengine)
https://www.volcengine.com/docs/82379/1263482

:::tip

**We support ALL Volcengine models including Chat and Embeddings, just set `model=volcengine/<any-model-on-volcengine>` as a prefix when sending litellm requests**

:::

## API Key
```python
# env variable
os.environ['VOLCENGINE_API_KEY']
# or
os.environ['ARK_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['VOLCENGINE_API_KEY'] = ""
response = completion(
    model="volcengine/<OUR_ENDPOINT_ID>",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    temperature=0.2,        # optional
    top_p=0.9,              # optional
    frequency_penalty=0.1,  # optional
    presence_penalty=0.1,   # optional
    max_tokens=10,          # optional
    stop=["\n\n"],          # optional
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['VOLCENGINE_API_KEY'] = ""
response = completion(
    model="volcengine/<OUR_ENDPOINT_ID>",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    stream=True,
    temperature=0.2,        # optional
    top_p=0.9,              # optional
    frequency_penalty=0.1,  # optional
    presence_penalty=0.1,   # optional
    max_tokens=10,          # optional
    stop=["\n\n"],          # optional
)

for chunk in response:
    print(chunk)
```

## Sample Usage - Embedding
```python
from litellm import embedding
import os

os.environ['VOLCENGINE_API_KEY'] = ""
response = embedding(
    model="volcengine/doubao-embedding-text-240715",
    input=["hello world", "good morning"]
)
print(response)
```

### Supported Embedding Models
- `doubao-embedding-large` (2048 dimensions)
- `doubao-embedding-large-text-250515` (2048 dimensions)
- `doubao-embedding-large-text-240915` (4096 dimensions)
- `doubao-embedding` (2560 dimensions) 
- `doubao-embedding-text-240715` (2560 dimensions)

### Embedding Parameters
```python
from litellm import embedding

response = embedding(
    model="volcengine/doubao-embedding-text-240715",
    input=["sample text"],
    encoding_format="float",  # optional: "float" (default), "base64"
    user="user-123",          # optional: user identifier for tracking
)
```

## Supported Chat Models

| Model ID | Context Window |
|---|---|
| `doubao-seed-2-0-pro-260215` | 256,000 |
| `kimi-k2-5-260127` | 256,000 |
| `glm-4-7-251222` | 200,000 |
| `deepseek-v3-2-251201` | 128,000 |

All chat models support function calling and tool choice.

## Supported Models - 💥 ALL Volcengine Models Supported!
We support ALL `volcengine` models for both chat completions and embeddings:
- **Chat Models**: Set `volcengine/<model-id>` as a prefix when sending completion requests (e.g., `volcengine/doubao-seed-2-0-pro-260215`)
- **Embedding Models**: Use the specific model names listed above (e.g., `volcengine/doubao-embedding-text-240715`)

## Sample Usage - LiteLLM Proxy

### Config.yaml setting

```yaml
model_list:
  # Chat model
  - model_name: volcengine-model
    litellm_params:
      model: volcengine/<OUR_ENDPOINT_ID>
      api_key: os.environ/VOLCENGINE_API_KEY
  # Embedding model
  - model_name: volcengine-embedding
    litellm_params:
      model: volcengine/doubao-embedding-text-240715
      api_key: os.environ/VOLCENGINE_API_KEY
```

### Send Request

#### Chat Completion
```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "volcengine-model",
    "messages": [
        {
        "role": "user",
        "content": "here is my api key. openai_api_key=sk-1234"
        }
    ]
}'
```

#### Embedding
```shell
curl --location 'http://localhost:4000/embeddings' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "volcengine-embedding",
    "input": ["hello world", "good morning"]
}'
```

## Related Providers

| Provider | Prefix | Description | Docs |
|---|---|---|---|
| [Volcengine Plan](./volcano_plan.md) | `volcengine_plan/` | Coding-optimized models via `/api/coding/v3` endpoint. Shares API key with Volcengine. | [volcano_plan](./volcano_plan.md) |
| [BytePlus](./byteplus.md) | `byteplus/` | International version of Volcengine for Southeast Asia. | [byteplus](./byteplus.md) |
| [BytePlus Plan](./byteplus_plan.md) | `byteplus_plan/` | Coding-optimized models on BytePlus international endpoint. | [byteplus_plan](./byteplus_plan.md) |