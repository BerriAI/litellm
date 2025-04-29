# Volcano Engine (Volcengine)
https://www.volcengine.com/docs/82379/1263482

:::tip

**We support ALL Volcengine NIM models, just set `model=volcengine/<any-model-on-volcengine>` as a prefix when sending litellm requests**

:::

## API Key
```python
# env variable
os.environ['VOLCENGINE_API_KEY']
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


## Supported Models - 💥 ALL Volcengine NIM Models Supported!
We support ALL `volcengine` models, just set `volcengine/<OUR_ENDPOINT_ID>` as a prefix when sending completion requests

## Sample Usage - LiteLLM Proxy

### Config.yaml setting

```yaml
model_list:
  - model_name: volcengine-model
    litellm_params:
      model: volcengine/<OUR_ENDPOINT_ID>
      api_key: os.environ/VOLCENGINE_API_KEY
```

### Send Request

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