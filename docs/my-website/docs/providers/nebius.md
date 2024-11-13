import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Nebius
https://docs.nebius.com/studio/inference/api/

:::tip

**We support ALL Nebius models, just set `model=nebius/<any-model-on-nebius>` as a prefix when sending litellm requests. For the complete supported model list, visit https://docs.nebius.com/studio/inference/models/ **

:::

## API Key
```python
# env variable
os.environ['NEBIUS_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['NEBIUS_API_KEY'] = ""
response = completion(
    model="nebius/meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Tell me about nebius.ai",
        }
    ],
    max_tokens=100,
    temperature=0.2,
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['NEBIUS_API_KEY'] = ""
response = completion(
    model="nebius/meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Tell me about nebius.ai",
        }
    ],
    stream=True,
    max_tokens=100,
)

for chunk in response:
    print(chunk)
```

## Usage with LiteLLM Proxy Server

Here's how to call a Nebius model with the LiteLLM Proxy Server

1. Modify the config.yaml

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: nebius/<your-model-name>  # add nebius/ prefix to route as Nebius provider
        api_key: api-key                 # api key to send your model
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
```
