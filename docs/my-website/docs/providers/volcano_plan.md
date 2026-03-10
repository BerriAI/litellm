# Volcengine Plan (volcengine_plan)

Volcengine Plan provides access to coding-optimized models through the `/api/coding/v3` endpoint.

:::tip

**Set `model=volcengine_plan/<model-id>` as a prefix when sending litellm requests.**

Volcengine Plan shares API keys with the standard [Volcengine](./volcano.md) provider (`VOLCENGINE_API_KEY` / `ARK_API_KEY`).

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
    model="volcengine_plan/ark-code-latest",
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

os.environ['VOLCENGINE_API_KEY'] = ""
response = completion(
    model="volcengine_plan/ark-code-latest",
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

os.environ['VOLCENGINE_API_KEY'] = ""
response = completion(
    model="volcengine_plan/doubao-seed-2.0-pro",
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
| `doubao-seed-code` | doubao-seed-code | 256,000 |
| `glm-4.7` | glm-4.7 | 200,000 |
| `deepseek-v3.2` | deepseek-v3.2 | 128,000 |
| `doubao-seed-2.0-code` | Doubao-Seed-2.0-Code | 256,000 |
| `doubao-seed-2.0-pro` | Doubao-Seed-2.0-pro | 256,000 |
| `doubao-seed-2.0-lite` | Doubao-Seed-2.0-lite | 256,000 |
| `minimax-m2.5` | MiniMax-M2.5 | 200,000 |
| `kimi-k2.5` | kimi-k2.5 | 256,000 |

All models support function calling and tool choice.

## Sample Usage - LiteLLM Proxy

### Config.yaml setting

```yaml
model_list:
  - model_name: volcengine-plan-code
    litellm_params:
      model: volcengine_plan/ark-code-latest
      api_key: os.environ/VOLCENGINE_API_KEY
  - model_name: volcengine-plan-pro
    litellm_params:
      model: volcengine_plan/doubao-seed-2.0-pro
      api_key: os.environ/VOLCENGINE_API_KEY
```

### Send Request

```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "volcengine-plan-code",
    "messages": [
        {
            "role": "user",
            "content": "Write a binary search function"
        }
    ]
}'
```
