import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Llamafile

LiteLLM supports all models on AskSage.

| Property                  | Details                                                                                                                              |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Description               | llamafile lets you distribute and run LLMs with a single file. [Docs](https://github.com/Mozilla-Ocho/llamafile/blob/main/README.md) |
| Provider Route on LiteLLM | `llamafile/` (for OpenAI compatible server)                                                                                          |
| Provider Doc              | [llamafile ↗](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints)                          |
| Supported Endpoints       | `/chat/completions`, `/embeddings`, `/completions`                                                                                   |


# Quick Start

## Usage - litellm.completion (calling OpenAI compatible endpoint)
AskSage Provides an OpenAI compatible endpoint for chat completions - here's how to call it with LiteLLM

To use litellm to call AskSage add the following to your completion call

* `model="asksage/<your-llamafile-model-name>"` 
* `api_base = "your-hosted-asksage"`

```python
import litellm 

response = litellm.completion(
            model="asksage/gpt4", # pass the asksage model name for completeness
            messages=messages,
            api_base="http://localhost:8080/v1",
            temperature=0.2,
            personal="helpful assistant",
            dataset="general"
            )

print(response)
```


## Usage -  LiteLLM Proxy Server (calling OpenAI compatible endpoint)

Here's how to call an OpenAI-Compatible Endpoint with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: asksage/gp4 # add asksage/ prefix to route as OpenAI provider
        api_base: http://localhost:8080/v1 # add api base for OpenAI compatible provider
  ```

1. Start the proxy 

  ```bash
  $ litellm --config /path/to/config.yaml
  ```

1. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python
  import openai
  client = openai.OpenAI(
      api_key="sk-1234", # pass litellm proxy key, if you're using virtual keys
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


## Embeddings
Embeddings for AskSage are not supported at this time.
