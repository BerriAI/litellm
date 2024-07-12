# [BETA] Anthropic `/v1/messages`

Call 100+ LLMs in the Anthropic format. 


1. Setup config.yaml 

```yaml
model_list:
  - model_name: my-test-model
    litellm_params:
      model: gpt-3.5-turbo
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/v1/messages' \
-H 'x-api-key: sk-1234' \
-H 'content-type: application/json' \
-D '{
    "model": "my-test-model",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, world"}
    ]
}'
```

## Test with Anthropic SDK 

```python
import os
from anthropic import Anthropic

client = Anthropic(api_key="sk-1234", base_url="http://0.0.0.0:4000") # 👈 CONNECT TO PROXY

message = client.messages.create(
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude",
        }
    ],
    model="my-test-model", # 👈 set 'model_name'
)
print(message.content)
```