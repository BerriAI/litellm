import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Weights & Biases Inference
https://weave-docs.wandb.ai/quickstart-inference

:::tip

Litellm provides support to all models from W&B Inference service. To use a model, set `model=wandb/<any-model-on-wandb-inference-dashboard>` as a prefix for litellm requests. The full list of supported models is provided at https://docs.wandb.ai/guides/inference/models/

:::

## API Key

You can get an API key for W&B Inference at - https://wandb.ai/authorize

```python
import os
# env variable
os.environ['WANDB_API_KEY']
```

## Sample Usage: Text Generation
```python
from litellm import completion
import os

os.environ['WANDB_API_KEY'] = "insert-your-wandb-api-key"
response = completion(
    model="wandb/Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[
        {
            "role": "user",
            "content": "What character was Wall-e in love with?",
        }
    ],
    max_tokens=10,
    response_format={ "type": "json_object" },
    seed=123,
    temperature=0.6,  # either set temperature or `top_p`
    top_p=0.01,  # to get as deterministic results as possible
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['WANDB_API_KEY'] = ""
response = completion(
    model="wandb/Qwen/Qwen3-235B-A22B-Instruct-2507",
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
    temperature=0.6,  # either set temperature or `top_p`
    top_p=0.01,  # to get as deterministic results as possible
)

for chunk in response:
    print(chunk)
```

:::tip

The above examples may not work if the model has been taken offline. Check the full list of available models at https://docs.wandb.ai/guides/inference/models/.

:::

## Usage with LiteLLM Proxy Server

Here's how to call a W&B Inference model with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: wandb/<your-model-name>  # add wandb/ prefix to use W&B Inference as provider
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

The W&B Inference provider supports the following parameters:

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
| response_format | object | Format of the response, e.g., `{"type": "json"}` |
| seed | integer | Sampling seed for deterministic results |
| stop | string/array | Sequences where the API will stop generating tokens |
| stream | boolean | Whether to stream the response |
| temperature | number | Controls randomness (0-2) |
| top_p | number | Controls nucleus sampling |


## Error Handling

The integration uses the standard LiteLLM error handling. Further, here's a list of commonly encountered errors with the W&B Inference API - 

| Error Code | Message | Cause | Solution |
| ---------- | ------- | ----- | -------- |
| 401 | Authentication failed | Your authentication credentials are incorrect or your W&B project entity and/or name are incorrect. | Ensure you're using the correct API key and that your W&B project name and entity are correct. |
| 403 | Country, region, or territory not supported | Accessing the API from an unsupported location. | Please see [Geographic restrictions](https://docs.wandb.ai/guides/inference/usage-limits/#geographic-restrictions) |
| 429 | Concurrency limit reached for requests | Too many concurrent requests. | Reduce the number of concurrent requests or increase your limits. For more information, see [Usage information and limits](https://docs.wandb.ai/guides/inference/usage-limits/). |
| 429 | You exceeded your current quota, please check your plan and billing details | Out of credits or reached monthly spending cap. | Get more credits or increase your limits. For more information, see [Usage information and limits](https://docs.wandb.ai/guides/inference/usage-limits/). |
| 429 | W&B Inference isn't available for personal accounts. | Switch to a non-personal account.  | Follow [the instructions below](#error-429-personal-entities-unsupported) for a work around. |
| 500 | The server had an error while processing your request | Internal server error. | Retry after a brief wait and contact support if it persists. |
| 503 | The engine is currently overloaded, please try again later | Server is experiencing high traffic. | Retry your request after a short delay. |


### Error 429: Personal entities unsupported

The user is on a personal account, which doesn't have access to W&B Inference. If one isn't available, create a Team to create a non-personal account. 

Once done, add the `openai-project` header to your request as shown below:

```python
response = completion(
    model="...",
    extra_headers={"openai-project": "team_name/project_name"},
    ...
```

For more information, see [Personal entities unsupported](https://docs.wandb.ai/guides/inference/usage-limits/#personal-entities-unsupported).

You can find more ways of using custom headers with LiteLLM here - https://docs.litellm.ai/docs/proxy/request_headers.
