import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Nvidia NIM
https://docs.api.nvidia.com/nim/reference/

:::tip

**We support ALL Nvidia NIM models, just set `model=nvidia_nim/<any-model-on-nvidia_nim>` as a prefix when sending litellm requests**

:::

| Property | Details |
|-------|-------|
| Description | Nvidia NIM is a platform that provides a simple API for deploying and using AI models. LiteLLM supports all models from [Nvidia NIM](https://developer.nvidia.com/nim/) |
| Provider Route on LiteLLM | `nvidia_nim/` |
| Provider Doc | [Nvidia NIM Docs ↗](https://developer.nvidia.com/nim/) |
| API Endpoint for Provider | https://integrate.api.nvidia.com/v1/ (chat/embeddings), https://ai.api.nvidia.com/v1/ (rerank) |
| Supported OpenAI Endpoints | `/chat/completions`, `/completions`, `/responses`, `/embeddings`, `/rerank` |

## API Key
```python
# env variable
os.environ['NVIDIA_NIM_API_KEY'] = ""
os.environ['NVIDIA_NIM_API_BASE'] = "" # [OPTIONAL] - default is https://integrate.api.nvidia.com/v1/
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['NVIDIA_NIM_API_KEY'] = ""
response = completion(
    model="nvidia_nim/meta/llama3-70b-instruct",
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

os.environ['NVIDIA_NIM_API_KEY'] = ""
response = completion(
    model="nvidia_nim/meta/llama3-70b-instruct",
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


## Usage - embedding

```python
import litellm
import os

response = litellm.embedding(
    model="nvidia_nim/nvidia/nv-embedqa-e5-v5",               # add `nvidia_nim/` prefix to model so litellm knows to route to Nvidia NIM
    input=["good morning from litellm"],
    encoding_format = "float", 
    user_id = "user-1234",

    # Nvidia NIM Specific Parameters
    input_type = "passage", # Optional
    truncate = "NONE" # Optional
)
print(response)
```


## **Usage - LiteLLM Proxy Server**

Here's how to call an Nvidia NIM Endpoint with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: nvidia_nim/<your-model-name>  # add nvidia_nim/ prefix to route as Nvidia NIM provider
        api_key: api-key                 # api key to send your model
       # api_base: "" # [OPTIONAL] - default is https://integrate.api.nvidia.com/v1/
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



## Supported Models - 💥 ALL Nvidia NIM Models Supported!
We support ALL `nvidia_nim` models, just set `nvidia_nim/` as a prefix when sending completion requests

| Model Name | Function Call |
|------------|---------------|
| nvidia/nemotron-4-340b-reward | `completion(model="nvidia_nim/nvidia/nemotron-4-340b-reward", messages)` |
| 01-ai/yi-large | `completion(model="nvidia_nim/01-ai/yi-large", messages)` |
| aisingapore/sea-lion-7b-instruct | `completion(model="nvidia_nim/aisingapore/sea-lion-7b-instruct", messages)` |
| databricks/dbrx-instruct | `completion(model="nvidia_nim/databricks/dbrx-instruct", messages)` |
| google/gemma-7b | `completion(model="nvidia_nim/google/gemma-7b", messages)` |
| google/gemma-2b | `completion(model="nvidia_nim/google/gemma-2b", messages)` |
| google/codegemma-1.1-7b | `completion(model="nvidia_nim/google/codegemma-1.1-7b", messages)` |
| google/codegemma-7b | `completion(model="nvidia_nim/google/codegemma-7b", messages)` |
| google/recurrentgemma-2b | `completion(model="nvidia_nim/google/recurrentgemma-2b", messages)` |
| ibm/granite-34b-code-instruct | `completion(model="nvidia_nim/ibm/granite-34b-code-instruct", messages)` |
| ibm/granite-8b-code-instruct | `completion(model="nvidia_nim/ibm/granite-8b-code-instruct", messages)` |
| mediatek/breeze-7b-instruct | `completion(model="nvidia_nim/mediatek/breeze-7b-instruct", messages)` |
| meta/codellama-70b | `completion(model="nvidia_nim/meta/codellama-70b", messages)` |
| meta/llama2-70b | `completion(model="nvidia_nim/meta/llama2-70b", messages)` |
| meta/llama3-8b | `completion(model="nvidia_nim/meta/llama3-8b", messages)` |
| meta/llama3-70b | `completion(model="nvidia_nim/meta/llama3-70b", messages)` |
| microsoft/phi-3-medium-4k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-medium-4k-instruct", messages)` |
| microsoft/phi-3-mini-128k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-mini-128k-instruct", messages)` |
| microsoft/phi-3-mini-4k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-mini-4k-instruct", messages)` |
| microsoft/phi-3-small-128k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-small-128k-instruct", messages)` |
| microsoft/phi-3-small-8k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-small-8k-instruct", messages)` |
| mistralai/codestral-22b-instruct-v0.1 | `completion(model="nvidia_nim/mistralai/codestral-22b-instruct-v0.1", messages)` |
| mistralai/mistral-7b-instruct | `completion(model="nvidia_nim/mistralai/mistral-7b-instruct", messages)` |
| mistralai/mistral-7b-instruct-v0.3 | `completion(model="nvidia_nim/mistralai/mistral-7b-instruct-v0.3", messages)` |
| mistralai/mixtral-8x7b-instruct | `completion(model="nvidia_nim/mistralai/mixtral-8x7b-instruct", messages)` |
| mistralai/mixtral-8x22b-instruct | `completion(model="nvidia_nim/mistralai/mixtral-8x22b-instruct", messages)` |
| mistralai/mistral-large | `completion(model="nvidia_nim/mistralai/mistral-large", messages)` |
| nvidia/nemotron-4-340b-instruct | `completion(model="nvidia_nim/nvidia/nemotron-4-340b-instruct", messages)` |
| seallms/seallm-7b-v2.5 | `completion(model="nvidia_nim/seallms/seallm-7b-v2.5", messages)` |
| snowflake/arctic | `completion(model="nvidia_nim/snowflake/arctic", messages)` |
| upstage/solar-10.7b-instruct | `completion(model="nvidia_nim/upstage/solar-10.7b-instruct", messages)` |