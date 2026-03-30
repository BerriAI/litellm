# Cortecs AI
LiteLLM supports all models available on Cortecs AI.

## Usage with LiteLLM Python SDK

```python
import os
from litellm import completion 

os.environ["CORTECS_API_KEY"] = "your-cortecs-api-key"

messages = [{"role": "user", "content": "Write a short poem"}]
response = completion(model="cortecs/glm-5", messages=messages)
print(response)
```

## Usage with LiteLLM Proxy 

### 1. Set Cortecs models in config.yaml

```yaml
model_list:
  - model_name: cortecs-model
    litellm_params:
      model: cortecs/glm-5
      api_key: "os.environ/CORTECS_API_KEY" # ensure you have `CORTECS_API_KEY` in your .env
```

### 2. Start proxy 

```bash
litellm --config config.yaml
```

### 3. Query proxy 

Assuming the proxy is running on [http://localhost:4000](http://localhost:4000):
```bash
curl http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_LITELLM_MASTER_KEY" \
  -d '{
    "model": "cortecs-model",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Write a short poem"
      }
    ]
  }'
```
`-H "Authorization: Bearer YOUR_LITELLM_MASTER_KEY" ` is only required if you have set a LiteLLM master key
