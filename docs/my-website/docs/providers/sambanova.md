import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Sambanova
https://community.sambanova.ai/t/create-chat-completion-api/

:::tip

**We support ALL Sambanova models, just set `model=sambanova/<any-model-on-sambanova>` as a prefix when sending litellm requests. For the complete supported model list, visit https://sambanova.ai/technology/models **

:::

## API Key
```python
# env variable
os.environ['SAMBANOVA_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['SAMBANOVA_API_KEY'] = ""
response = completion(
    model="sambanova/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What do you know about sambanova.ai",
        }
    ],
    max_tokens=10,
    response_format={ "type": "json_object" },
    seed=123,
    stop=["\n\n"],
    temperature=0.2,
    top_p=0.9,
    tool_choice="auto",
    tools=[],
    user="user",
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['SAMBANOVA_API_KEY'] = ""
response = completion(
    model="sambanova/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What do you know about sambanova.ai",
        }
    ],
    stream=True,
    max_tokens=10,
    response_format={ "type": "json_object" },
    seed=123,
    stop=["\n\n"],
    temperature=0.2,
    top_p=0.9,
    tool_choice="auto",
    tools=[],
    user="user",
)

for chunk in response:
    print(chunk)
```


## Usage with LiteLLM Proxy Server

Here's how to call a Sambanova model with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: sambanova/<your-model-name>  # add sambanova/ prefix to route as Sambanova provider
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
