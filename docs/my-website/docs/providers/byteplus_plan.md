# BytePlus Plan (byteplus_plan)

BytePlus Plan provides access to coding-optimized models on the BytePlus international endpoint via `/api/coding/v3`.

:::tip

**Set `model=byteplus_plan/<model-id>` as a prefix when sending litellm requests.**

BytePlus Plan shares API keys with the standard [BytePlus](./byteplus.md) provider (`BYTEPLUS_API_KEY`).

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
    model="byteplus_plan/ark-code-latest",
    messages=[
        {
            "role": "user",
            "content": "Write a Python function to sort a list",
        }
    ],
    max_tokens=1024,
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['BYTEPLUS_API_KEY'] = ""
response = completion(
    model="byteplus_plan/ark-code-latest",
    messages=[
        {
            "role": "user",
            "content": "Write a Python function to sort a list",
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
    model="byteplus_plan/kimi-k2.5",
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
| `ark-code-latest` | ark-code-latest | 256,000 |
| `bytedance-seed-code` | bytedance-seed-code | 256,000 |
| `glm-4.7` | glm-4.7 | 200,000 |
| `kimi-k2.5` | kimi-k2.5 | 256,000 |

All models support function calling and tool choice.

## Sample Usage - LiteLLM Proxy

### Config.yaml setting

```yaml
model_list:
  - model_name: byteplus-plan-code
    litellm_params:
      model: byteplus_plan/ark-code-latest
      api_key: os.environ/BYTEPLUS_API_KEY
  - model_name: byteplus-plan-kimi
    litellm_params:
      model: byteplus_plan/kimi-k2.5
      api_key: os.environ/BYTEPLUS_API_KEY
```

### Send Request

```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "byteplus-plan-code",
    "messages": [
        {
            "role": "user",
            "content": "Write a binary search function"
        }
    ]
}'
```
