import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Z.AI (Zhipu AI)
https://z.ai/

**We support Z.AI GLM text/chat models, just set `zai/` as a prefix when sending completion requests**

## API Key
```python
# env variable
os.environ['ZAI_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['ZAI_API_KEY'] = ""
response = completion(
    model="zai/glm-4.6",
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['ZAI_API_KEY'] = ""
response = completion(
    model="zai/glm-4.6",
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Supported Models

We support ALL Z.AI GLM models, just set `zai/` as a prefix when sending completion requests.

| Model Name | Function Call | Notes |
|------------|---------------|-------|
| glm-4.6 | `completion(model="zai/glm-4.6", messages)` | Latest flagship model, 200K context |
| glm-4.5 | `completion(model="zai/glm-4.5", messages)` | 128K context |
| glm-4.5v | `completion(model="zai/glm-4.5v", messages)` | Vision model |
| glm-4.5-x | `completion(model="zai/glm-4.5-x", messages)` | Premium tier |
| glm-4.5-air | `completion(model="zai/glm-4.5-air", messages)` | Lightweight |
| glm-4.5-airx | `completion(model="zai/glm-4.5-airx", messages)` | Fast lightweight |
| glm-4-32b-0414-128k | `completion(model="zai/glm-4-32b-0414-128k", messages)` | 32B parameter model |
| glm-4.5-flash | `completion(model="zai/glm-4.5-flash", messages)` | **FREE tier** |

## Model Pricing

| Model | Input ($/1M tokens) | Output ($/1M tokens) | Context Window |
|-------|---------------------|----------------------|----------------|
| glm-4.6 | $0.60 | $2.20 | 200K |
| glm-4.5 | $0.60 | $2.20 | 128K |
| glm-4.5v | $0.60 | $1.80 | 128K |
| glm-4.5-x | $2.20 | $8.90 | 128K |
| glm-4.5-air | $0.20 | $1.10 | 128K |
| glm-4.5-airx | $1.10 | $4.50 | 128K |
| glm-4-32b-0414-128k | $0.10 | $0.10 | 128K |
| glm-4.5-flash | **FREE** | **FREE** | 128K |

## Using with LiteLLM Proxy

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ['ZAI_API_KEY'] = ""
response = completion(
    model="zai/glm-4.6",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: glm-4.6
    litellm_params:
        model: zai/glm-4.6
        api_key: os.environ/ZAI_API_KEY
  - model_name: glm-4.5-flash  # Free tier
    litellm_params:
        model: zai/glm-4.5-flash
        api_key: os.environ/ZAI_API_KEY
```

2. Run proxy

```bash
litellm --config config.yaml
```

3. Test it!

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "glm-4.6",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
}'
```

</TabItem>
</Tabs>
