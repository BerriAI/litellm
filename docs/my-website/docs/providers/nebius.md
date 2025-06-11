import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Nebius AI Studio
https://docs.nebius.com/studio/inference/quickstart

:::tip

**Litellm provides support to all models from Nebius AI Studio. To use a model, set `model=nebius/<any-model-on-nebius-ai-studio>` as a prefix for litellm requests. The full list of supported models is provided at https://studio.nebius.ai/ **

:::

## API Key
```python
import os
# env variable
os.environ['NEBIUS_API_KEY']
```

## Sample Usage: Text Generation
```python
from litellm import completion
import os

os.environ['NEBIUS_API_KEY'] = "insert-your-nebius-ai-studio-api-key"
response = completion(
    model="nebius/Qwen/Qwen3-235B-A22B",
    messages=[
        {
            "role": "user",
            "content": "What character was Wall-e in love with?",
        }
    ],
    max_tokens=10,
    response_format={ "type": "json_object" },
    seed=123,
    stop=["\n\n"],
    temperature=0.6,  # either set temperature or `top_p`
    top_p=0.01,  # to get as deterministic results as possible
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

os.environ['NEBIUS_API_KEY'] = ""
response = completion(
    model="nebius/Qwen/Qwen3-235B-A22B",
    messages=[
        {
            "role": "user",
            "content": "What character was Wall-e in love with?",
        }
    ],
    stream=True,
    max_tokens=10,
    response_format={ "type": "json_object" },
    seed=123,
    stop=["\n\n"],
    temperature=0.6,  # either set temperature or `top_p`
    top_p=0.01,  # to get as deterministic results as possible
    tool_choice="auto",
    tools=[],
    user="user",
)

for chunk in response:
    print(chunk)
```

## Sample Usage - Embedding
```python
from litellm import embedding
import os

os.environ['NEBIUS_API_KEY'] = ""
response = embedding(
    model="nebius/BAAI/bge-en-icl",
    input=["What character was Wall-e in love with?"],
)
print(response)
```


## Usage with LiteLLM Proxy Server

Here's how to call a Nebius AI Studio model with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: nebius/<your-model-name>  # add nebius/ prefix to use Nebius AI Studio as provider
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
      api_key="litellm-proxy-key",             # pass litellm proxy key, if you're using virtual keys
      base_url="http://0.0.0.0:4000" # litellm-proxy-base url
  )

  response = client.chat.completions.create(
      model="my-model",
      messages = [
          {
              "role": "user",
              "content": "What character was Wall-e in love with?"
          }
      ],
  )

  print(response)
  ```
  </TabItem>

  <TabItem value="curl" label="curl">

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: litellm-proxy-key' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "my-model",
      "messages": [
          {
          "role": "user",
          "content": "What character was Wall-e in love with?"
          }
      ],
  }'
  ```
  </TabItem>

  </Tabs>

## Supported Parameters

The Nebius provider supports the following parameters:

### Chat Completion Parameters

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| frequency_penalty | number | Penalizes new tokens based on their frequency in the text |
| function_call | string/object | Controls how the model calls functions |
| functions | array | List of functions for which the model may generate JSON inputs |
| logit_bias | map | Modifies the likelihood of specified tokens |
| max_tokens | integer | Maximum number of tokens to generate |
| n | integer | Number of completions to generate |
| presence_penalty | number | Penalizes tokens based on if they appear in the text so far |
| response_format | object | Format of the response, e.g., {"type": "json"} |
| seed | integer | Sampling seed for deterministic results |
| stop | string/array | Sequences where the API will stop generating tokens |
| stream | boolean | Whether to stream the response |
| temperature | number | Controls randomness (0-2) |
| top_p | number | Controls nucleus sampling |
| tool_choice | string/object | Controls which (if any) function to call |
| tools | array | List of tools the model can use |
| user | string | User identifier |

### Embedding Parameters

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| input | string/array | Text to embed |
| user | string | User identifier |

## Error Handling

The integration uses the standard LiteLLM error handling. Common errors include:

- **Authentication Error**: Check your API key
- **Model Not Found**: Ensure you're using a valid model name
- **Rate Limit Error**: You've exceeded your rate limits
- **Timeout Error**: Request took too long to complete
