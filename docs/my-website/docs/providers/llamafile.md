import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Llamafile

LiteLLM supports all models on Llamafile.

| Property                  | Details                                                                                                                              |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Description               | llamafile lets you distribute and run LLMs with a single file. [Docs](https://github.com/Mozilla-Ocho/llamafile/blob/main/README.md) |
| Provider Route on LiteLLM | `llamafile/` (for OpenAI compatible server)                                                                                          |
| Provider Doc              | [llamafile ↗](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints)                          |
| Supported Endpoints       | `/chat/completions`, `/embeddings`, `/completions`                                                                                   |


# Quick Start

## Usage - litellm.completion (calling OpenAI compatible endpoint)
llamafile Provides an OpenAI compatible endpoint for chat completions - here's how to call it with LiteLLM

To use litellm to call llamafile add the following to your completion call

* `model="llamafile/<your-llamafile-model-name>"` 
* `api_base = "your-hosted-llamafile"`

```python
import litellm 

response = litellm.completion(
            model="llamafile/mistralai/mistral-7b-instruct-v0.2", # pass the llamafile model name for completeness
            messages=messages,
            api_base="http://localhost:8080/v1",
            temperature=0.2,
            max_tokens=80)

print(response)
```


## Usage -  LiteLLM Proxy Server (calling OpenAI compatible endpoint)

Here's how to call an OpenAI-Compatible Endpoint with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: llamafile/mistralai/mistral-7b-instruct-v0.2 # add llamafile/ prefix to route as OpenAI provider
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

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import embedding   
import os

os.environ["LLAMAFILE_API_BASE"] = "http://localhost:8080/v1"


embedding = embedding(model="llamafile/sentence-transformers/all-MiniLM-L6-v2", input=["Hello world"])

print(embedding)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
    - model_name: my-model
      litellm_params:
        model: llamafile/sentence-transformers/all-MiniLM-L6-v2 # add llamafile/ prefix to route as OpenAI provider
        api_base: http://localhost:8080/v1 # add api base for OpenAI compatible provider
```

1. Start the proxy 

```bash
$ litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

1. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/embeddings' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{"input": ["hello world"], "model": "my-model"}'
```

[See OpenAI SDK/Langchain/etc. examples](../proxy/user_keys.md#embeddings)

</TabItem>
</Tabs>