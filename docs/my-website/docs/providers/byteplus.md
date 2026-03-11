# BytePlus

BytePlus is the international version of Volcengine (ByteDance), serving the Southeast Asia region via the `/api/v3` endpoint.

:::tip

**Set `model=byteplus/<model-id>` as a prefix when sending litellm requests.**

See also: [BytePlus Plan](./byteplus_plan.md) for coding-optimized models.

:::

## API Key
```python
# env variable
os.environ['BYTEPLUS_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['BYTEPLUS_API_KEY'] = ""
response = completion(
    model="byteplus/seed-2-0-mini-260215",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Singapore today?",
        }
    ],
    temperature=0.2,
    max_tokens=512,
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['BYTEPLUS_API_KEY'] = ""
response = completion(
    model="byteplus/seed-2-0-mini-260215",
    messages=[
        {
            "role": "user",
            "content": "Explain recursion step by step",
        }
    ],
    stream=True,
    max_tokens=1024,
)

for chunk in response:
    print(chunk)
```

## Sample Usage - Thinking Parameter
```python
from litellm import completion
import os

os.environ['BYTEPLUS_API_KEY'] = ""
response = completion(
    model="byteplus/kimi-k2-5-260127",
    messages=[
        {
            "role": "user",
            "content": "Explain the difference between a stack and a queue",
        }
    ],
    thinking={"type": "enabled"},  # "enabled", "disabled", or "auto"
    max_tokens=2048,
)
print(response)
```

## Supported Models

| Model ID | Display Name | Context Window |
|---|---|---|
| `seed-2-0-mini-260215` | seed-2-0-mini-260215 | 256,000 |
| `kimi-k2-5-260127` | kimi-k2-5-260127 | 256,000 |
| `glm-4-7-251222` | glm-4-7-251222 | 200,000 |

All models support function calling and tool choice.

## Sample Usage - LiteLLM Proxy

### Config.yaml setting

```yaml
model_list:
  - model_name: byteplus-chat
    litellm_params:
      model: byteplus/seed-2-0-mini-260215
      api_key: os.environ/BYTEPLUS_API_KEY
```

### Send Request

```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "byteplus-chat",
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
}'
```
