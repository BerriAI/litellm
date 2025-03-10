import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Nvidia NIM
https://docs.api.nvidia.com/nim/reference/

:::tip

**We support ALL Nvidia NIM models, just set `model=nvidia_nim/<any-model-on-nvidia_nim>` as a prefix when sending litellm requests**

:::

## API Key
```python
# env variable
os.environ['NVIDIA_NIM_API_KEY']
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

## **Function/Tool Calling**

```python
from litellm import completion

# set env
os.environ['NVIDIA_API_KEY'] = ""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

response = completion(
    model="nvidia/meta/llama-3.1-70b-instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
# Add any assertions, here to check response args
print(response)
assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
assert isinstance(
    response.choices[0].message.tool_calls[0].function.arguments, str
)

```

### Forcing Tool Use

If you want LLM to use a specific tool to answer the user's question

You can do this by specifying the tool in the `tool_choice` field like so:

```python
response = completion(
    os.environ['NVIDIA_API_KEY'] = ""
    messages=messages,
    tools=tools,
    tool_choice={"type": "tool", "name": "get_weather"},
)
```

## Usage - Vision 

```python
from litellm import completion

# set env
os.environ['NVIDIA_API_KEY'] = ""

def encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_path = "nvidia-picasso.jpg"
# Getting the base64 string
base64_image = encode_image(image_path)
response = litellm.completion(
    model="nvidia/microsoft/phi-3-vision-128k-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Whats in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64_image
                    },
                },
            ],
        }
    ],
)
print(f"\nResponse: {response}")
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
We support ALL `nvidia_nim` models, just set `nvidia_nim/` or `nvidia` as a prefix when sending completion requests

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
| google/gemma-2-2b-it | `completion(model="nvidia_nim/google/gemma-2-2b-it", messages)` |
| google/gemma-2-9b-it | `completion(model="nvidia_nim/google/gemma-2-9b-it", messages)` |
| google/gemma-2-27b-it | `completion(model="nvidia_nim/google/gemma-2-27b-it", messages)` |
| ibm/granite-34b-code-instruct | `completion(model="nvidia_nim/ibm/granite-34b-code-instruct", messages)` |
| ibm/granite-8b-code-instruct | `completion(model="nvidia_nim/ibm/granite-8b-code-instruct", messages)` |
| ibm/granite-3.0-8b-instruct | `completion(model="nvidia_nim/ibm/granite-3.0-8b-instruct", messages)` |
| ibm/granite-3.0-3b-a800m-instruct | `completion(model="nvidia_nim/ibm/granite-3.0-3b-a800m-instruct", messages)` |
| mediatek/breeze-7b-instruct | `completion(model="nvidia_nim/mediatek/breeze-7b-instruct", messages)` |
| meta/codellama-70b | `completion(model="nvidia_nim/meta/codellama-70b", messages)` |
| meta/llama2-70b | `completion(model="nvidia_nim/meta/llama2-70b", messages)` |
| meta/llama3-8b-instruct | `completion(model="nvidia_nim/meta/llama3-8b-instruct", messages)` |
| meta/llama3-70b-instruct | `completion(model="nvidia_nim/meta/llama3-70b-instruct", messages)` |
| meta/llama-3.1-8b-instruct | `completion(model="nvidia_nim/meta/llama-3.1-8b-instruct", messages)` |
| meta/llama-3.1-70b-instruct | `completion(model="nvidia_nim/meta/llama-3.1-70b-instruct", messages)` |
| meta/llama-3.1-405b-instruct | `completion(model="nvidia_nim/meta/llama-3.1-405b-instruct", messages)` |
| meta/llama-3.2-1b-instruct | `completion(model="nvidia_nim/meta/llama-3.2-1b-instruct", messages)` |
| meta/llama-3.2-3b-instruct | `completion(model="nvidia_nim/meta/llama-3.2-3b-instruct", messages)` |
| meta/llama-3.2-11b-vision-instruct | `completion(model="nvidia_nim/meta/llama-3.2-11b-vision-instruct", messages)` |
| meta/llama-3.2-90b-vision-instruct | `completion(model="nvidia_nim/meta/llama-3.2-90b-vision-instruct", messages)` |
| meta/llama-3.3-70b-instruct | `completion(model="nvidia_nim/meta/llama-3.3-70b-instruct", messages)` |
| microsoft/phi-3-medium-4k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-medium-4k-instruct", messages)` |
| microsoft/phi-3-mini-128k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-mini-128k-instruct", messages)` |
| microsoft/phi-3-mini-4k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-mini-4k-instruct", messages)` |
| microsoft/phi-3-small-128k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-small-128k-instruct", messages)` |
| microsoft/phi-3-small-8k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-small-8k-instruct", messages)` |
| microsoft/phi-3-medium-128k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-medium-128k-instruct", messages)` |
| microsoft/phi-3-vision-128k-instruct | `completion(model="nvidia_nim/microsoft/phi-3-vision-128k-instruct", messages)` |
| microsoft/phi-3.5-mini-instruct | `completion(model="nvidia_nim/microsoft/phi-3.5-mini-instruct", messages)` |
| microsoft/phi-3.5-moe-instruct | `completion(model="nvidia_nim/microsoft/phi-3.5-moe-instruct", messages)` |
| microsoft/phi-3.5-vision-instruct | `completion(model="nvidia_nim/microsoft/phi-3.5-vision-instruct", messages)` |
| mistralai/codestral-22b-instruct-v0.1 | `completion(model="nvidia_nim/mistralai/codestral-22b-instruct-v0.1", messages)` |
| mistralai/mistral-7b-instruct | `completion(model="nvidia_nim/mistralai/mistral-7b-instruct", messages)` |
| mistralai/mistral-7b-instruct-v0.2 | `completion(model="nvidia_nim/mistralai/mistral-7b-instruct-v0.2", messages)` |
| mistralai/mistral-7b-instruct-v0.3 | `completion(model="nvidia_nim/mistralai/mistral-7b-instruct-v0.3", messages)` |
| mistralai/mixtral-8x7b-instruct | `completion(model="nvidia_nim/mistralai/mixtral-8x7b-instruct", messages)` |
| mistralai/mixtral-8x22b-instruct | `completion(model="nvidia_nim/mistralai/mixtral-8x22b-instruct", messages)` |
| mistralai/mistral-large | `completion(model="nvidia_nim/mistralai/mistral-large", messages)` |
| mistralai/mistral-large-2-instruct | `completion(model="nvidia_nim/mistralai/mistral-large-2-instruct", messages)` |
| mistralai/mamba-codestral-7b-v0.1 | `completion(model="nvidia_nim/mistralai/mamba-codestral-7b-v0.1", messages)` |
| mistralai/mathstral-7b-v0.1 | `completion(model="nvidia_nim/mistralai/mathstral-7b-v0.1", messages)` |
| nvidia/nemotron-4-340b-instruct | `completion(model="nvidia_nim/nvidia/nemotron-4-340b-instruct", messages)` |
| nvidia/nemotron-mini-4b-instruct | `completion(model="nvidia_nim/nvidia/nemotron-mini-4b-instruct", messages)` |
| nvidia/nemotron-4-mini-hindi-4b-instruct | `completion(model="nvidia_nim/nvidia/nemotron-4-mini-hindi-4b-instruct", messages)` |
| nvidia/usdcode-llama3-70b-instruct | `completion(model="nvidia_nim/nvidia/usdcode-llama3-70b-instruct", messages)` |
| nvidia/usdcode-llama-3.1-70b-instruct | `completion(model="nvidia_nim/nvidia/usdcode-llama-3.1-70b-instruct", messages)` |
| nvidia/llama-3.1-nemotron-51b-instruct | `completion(model="nvidia_nim/nvidia/llama-3.1-nemotron-51b-instruct", messages)` |
| nvidia/llama-3.1-nemotron-70b-instruct | `completion(model="nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct", messages)` |
| nvidia/llama-3.1-nemotron-70b-reward | `completion(model="nvidia_nim/nvidia/llama-3.1-nemotron-70b-reward", messages)` |
| nvidia/mistral-nemo-12b-instruct | `completion(model="nvidia_nim/nv-mistralai/mistral-nemo-12b-instruct", messages)` |
| nvidia/mistral-nemo-minitron-8b-8k-instruct | `completion(model="nvidia_nim/nvidia/mistral-nemo-minitron-8b-8k-instruct", messages)` |
| nvidia/vila | `completion(model="nvidia_nim/nvidia/vila", messages)` |
| seallms/seallm-7b-v2.5 | `completion(model="nvidia_nim/seallms/seallm-7b-v2.5", messages)` |
| snowflake/arctic | `completion(model="nvidia_nim/snowflake/arctic", messages)` |
| upstage/solar-10.7b-instruct | `completion(model="nvidia_nim/upstage/solar-10.7b-instruct", messages)` |
| writer/palmyra-med-70b-32k | `completion(model="nvidia_nim/writer/palmyra-med-70b-32k", messages)` |
| writer/palmyra-med-70b | `completion(model="nvidia_nim/writer/palmyra-med-70b", messages)` |
| writer/palmyra-fin-70b-32k | `completion(model="nvidia_nim/writer/palmyra-fin-70b-32k", messages)` |
| deepseek-ai/deepseek-coder-6.7b-instruct | `completion(model="nvidia_nim/deepseek-ai/deepseek-coder-6.7b-instruct", messages)` |
| deepseek-ai/deepseek-r1 | `completion(model="nvidia_nim/deepseek-ai/deepseek-r1", messages)` |
| institute-of-science-tokyo/llama-3.1-swallow-8b-instruct-v0.1 | `completion(model="nvidia_nim/institute-of-science-tokyo/llama-3.1-swallow-8b-instruct-v0.1", messages)` |
| institute-of-science-tokyo/llama-3.1-swallow-70b-instruct-v0.1 | `completion(model="nvidia_nim/institute-of-science-tokyo/llama-3.1-swallow-70b-instruct-v0.1", messages)` |
| tokyotech-llm/llama-3-swallow-70b-instruct-v0.1 | `completion(model="nvidia_nim/tokyotech-llm/llama-3-swallow-70b-instruct-v0.1", messages)` |
| yentinglin/llama-3-taiwan-70b-instruct | `completion(model="nvidia_nim/yentinglin/llama-3-taiwan-70b-instruct", messages)` |
| zyphra/zamba2-7b-instruct | `completion(model="nvidia_nim/zyphra/zamba2-7b-instruct", messages)` |
| ai21labs/jamba-1.5-large-instruct | `completion(model="nvidia_nim/ai21labs/jamba-1.5-large-instruct", messages)` |
| ai21labs/jamba-1.5-mini-instruct | `completion(model="nvidia_nim/ai21labs/jamba-1.5-mini-instruct", messages)` |
| abacusai/dracarys-llama-3.1-70b-instruct | `completion(model="nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct", messages)` |
| qwen/qwen2-7b-instruct | `completion(model="nvidia_nim/qwen/qwen2-7b-instruct", messages)` |
| qwen/qwen2.5-coder-32b-instruct | `completion(model="nvidia_nim/qwen/qwen2.5-coder-32b-instruct", messages)` |
| qwen/qwen2.5-coder-7b-instruct | `completion(model="nvidia_nim/qwen/qwen2.5-coder-7b-instruct", messages)` |
| rakuten/rakutenai-7b-instruct | `completion(model="nvidia_nim/rakuten/rakutenai-7b-instruct", messages)` |
| rakuten/rakutenai-7b-chat | `completion(model="nvidia_nim/rakuten/rakutenai-7b-chat", messages)` |
| baichuan-inc/baichuan2-13b-chat | `completion(model="nvidia_nim/baichuan-inc/baichuan2-13b-chat", messages)` |
| thudm/chatglm3-6b | `completion(model="nvidia_nim/thudm/chatglm3-6b", messages)` |

### Embedding Models
| Model Name | Function Call |
|------------|---------------|
| snowflake/arctic-embed-l | `embedding(model="nvidia_nim/snowflake/arctic-embed-l", input)` |
| nvidia/nv-embed-v1 | `embedding(model="nvidia_nim/nvidia/nv-embed-v1", input)` |
| nvidia/nv-embedqa-mistral-7b-v2 | `embedding(model="nvidia_nim/nvidia/nv-embedqa-mistral-7b-v2", input)` |
| nvidia/nv-embedqa-e5-v5 | `embedding(model="nvidia_nim/nvidia/nv-embedqa-e5-v5", input)` |
| baai/bge-m3 | `embedding(model="nvidia_nim/baai/bge-m3", input)` |
| nvidia/embed-qa-4 | `embedding(model="nvidia_nim/nvidia/embed-qa-4", input)` |
| nvidia/llama-3.2-nv-embedqa-1b-v1 | `embedding(model="nvidia_nim/nvidia/llama-3.2-nv-embedqa-1b-v1", input)` |
| nvidia/llama-3.2-nv-embedqa-1b-v2 | `embedding(model="nvidia_nim/nvidia/llama-3.2-nv-embedqa-1b-v2", input)` |

> **NOTE: If you're using `nvidia` as a prefix, you can omit the `nvidia_nim/` prefix.
>Existing nvidia-nim implementations will continue to work as before.