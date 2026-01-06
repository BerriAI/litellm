import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Xiaomi MiMo
https://platform.xiaomimimo.com/#/docs

:::tip

**We support ALL Xiaomi MiMo models, just set `model=xiaomi_mimo/<any-model-on-xiaomi-mimo>` as a prefix when sending litellm requests**

:::

## API Key
```python
# env variable
os.environ['XIAOMI_MIMO_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['XIAOMI_MIMO_API_KEY'] = ""
response = completion(
    model="xiaomi_mimo/mimo-v2-flash",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    max_tokens=1024,
    temperature=0.3,
    top_p=0.95,
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['XIAOMI_MIMO_API_KEY'] = ""
response = completion(
    model="xiaomi_mimo/mimo-v2-flash",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    stream=True,
    max_tokens=1024,
    temperature=0.3,
    top_p=0.95,
)

for chunk in response:
    print(chunk)
```


## Usage with LiteLLM Proxy Server

Here's how to call a Xiaomi MiMo model with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: xiaomi_mimo/<your-model-name>  # add xiaomi_mimo/ prefix to route as Xiaomi MiMo provider
        api_key: api-key                      # api key to send your model
  ```


2. Start the proxy 

  ```bash
  $ litellm --config /path/to/config.yaml
  ```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python
  import openai
  client = openai.OpenAI(
      api_key="sk-1234",             # pass litellm proxy key, if you're using virtual keys
      base_url="http://0.0.0.0:4000" # litellm-proxy-base url
  )

  response = client.chat.completions.create(
      model="my-model",
      messages = [
          {
              "role": "user",
              "content": "what llm are you"
          }
      ],
  )

  print(response)
  ```
  </TabItem>

  <TabItem value="curl" label="curl">

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "my-model",
      "messages": [
          {
          "role": "user",
          "content": "what llm are you"
          }
      ],
  }'
  ```
  </TabItem>

  </Tabs>

## Supported Models

| Model Name | Usage |
|------------|-------|
| mimo-v2-flash | `completion(model="xiaomi_mimo/mimo-v2-flash", messages)` |
