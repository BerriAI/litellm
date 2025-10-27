import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Ollama 
LiteLLM supports all models from [Ollama](https://github.com/ollama/ollama)

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_Ollama.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

:::info 

We recommend using [ollama_chat](#using-ollama-apichat) for better responses.

:::

## Pre-requisites
Ensure you have your ollama server running

## Example usage
```python
from litellm import completion

response = completion(
    model="ollama/llama2", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://localhost:11434"
)
print(response)

```

## Example usage - Streaming
```python
from litellm import completion

response = completion(
    model="ollama/llama2", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://localhost:11434",
    stream=True
)
print(response)
for chunk in response:
    print(chunk['choices'][0]['delta'])

```

## Example usage - Streaming + Acompletion
Ensure you have async_generator installed for using ollama acompletion with streaming
```shell
pip install async_generator
```

```python
async def async_ollama():
    response = await litellm.acompletion(
        model="ollama/llama2", 
        messages=[{ "content": "what's the weather" ,"role": "user"}], 
        api_base="http://localhost:11434", 
        stream=True
    )
    async for chunk in response:
        print(chunk)

# call async_ollama
import asyncio
asyncio.run(async_ollama())

```

## Example Usage - JSON Mode
To use ollama JSON Mode pass `format="json"` to `litellm.completion()`

```python
from litellm import completion
response = completion(
  model="ollama/llama2",
  messages=[
      {
          "role": "user",
          "content": "respond in json, what's the weather"
      }
  ],
  max_tokens=10,
  format = "json"
)
```

## Example Usage - Tool Calling 

To use ollama tool calling, pass `tools=[{..}]` to `litellm.completion()` 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import litellm 

## [OPTIONAL] REGISTER MODEL - not all ollama models support function calling, litellm defaults to json mode tool calls if native tool calling not supported.

# litellm.register_model(model_cost={
#                 "ollama_chat/llama3.1": { 
#                   "supports_function_calling": true
#                 },
#             })

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
    }
  }
]

messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]


response = completion(
  model="ollama_chat/llama3.1",
  messages=messages,
  tools=tools
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml 

```yaml
model_list:
  - model_name: "llama3.1"             
    litellm_params:
      model: "ollama_chat/llama3.1"
      keep_alive: "8m" # Optional: Overrides default keep_alive, use -1 for Forever
    model_info:
      supports_function_calling: true
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "llama3.1",
  "messages": [
    {
      "role": "user",
      "content": "What'\''s the weather like in Boston today?"
    }
  ],
  "tools": [
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
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "stream": true
}'
```
</TabItem>
</Tabs>


## Using Ollama FIM on `/v1/completions`

LiteLLM supports calling Ollama's `/api/generate` endpoint on `/v1/completions` requests. 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm 
litellm._turn_on_debug() # turn on debug to see the request
from litellm import completion

response = completion(
    model="ollama/llama3.1",
    prompt="Hello, world!",
    api_base="http://localhost:11434"
)
print(response)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml 

```yaml
model_list:
  - model_name: "llama3.1"             
    litellm_params:
      model: "ollama/llama3.1"
      api_base: "http://localhost:11434"
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml --detailed_debug

# RUNNING ON http://0.0.0.0:4000 
```

3. Test it! 

```python
from openai import OpenAI

client = OpenAI(
    api_key="anything", # 👈 PROXY KEY (can be anything, if master_key not set)
    base_url="http://0.0.0.0:4000" # 👈 PROXY BASE URL
)

response = client.completions.create(
    model="ollama/llama3.1",
    prompt="Hello, world!",
    api_base="http://localhost:11434"
)
print(response)
```
</TabItem>
</Tabs>

## Using ollama `api/chat` 
In order to send ollama requests to `POST /api/chat` on your ollama server, set the model prefix to `ollama_chat`

```python
from litellm import completion

response = completion(
    model="ollama_chat/llama2", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
)
print(response)
```
## Ollama Models
Ollama supported models: https://github.com/ollama/ollama

| Model Name           | Function Call                                                                     |
|----------------------|-----------------------------------------------------------------------------------
| Mistral    | `completion(model='ollama/mistral', messages, api_base="http://localhost:11434", stream=True)` |
| Mistral-7B-Instruct-v0.1 | `completion(model='ollama/mistral-7B-Instruct-v0.1', messages, api_base="http://localhost:11434", stream=False)` |
| Mistral-7B-Instruct-v0.2 | `completion(model='ollama/mistral-7B-Instruct-v0.2', messages, api_base="http://localhost:11434", stream=False)` |
| Mixtral-8x7B-Instruct-v0.1 | `completion(model='ollama/mistral-8x7B-Instruct-v0.1', messages, api_base="http://localhost:11434", stream=False)` |
| Mixtral-8x22B-Instruct-v0.1 | `completion(model='ollama/mixtral-8x22B-Instruct-v0.1', messages, api_base="http://localhost:11434", stream=False)` |
| Llama2 7B            | `completion(model='ollama/llama2', messages, api_base="http://localhost:11434", stream=True)` | 
| Llama2 13B           | `completion(model='ollama/llama2:13b', messages, api_base="http://localhost:11434", stream=True)` | 
| Llama2 70B           | `completion(model='ollama/llama2:70b', messages, api_base="http://localhost:11434", stream=True)` | 
| Llama2 Uncensored    | `completion(model='ollama/llama2-uncensored', messages, api_base="http://localhost:11434", stream=True)` | 
| Code Llama    | `completion(model='ollama/codellama', messages, api_base="http://localhost:11434", stream=True)` | 
| Llama2 Uncensored    | `completion(model='ollama/llama2-uncensored', messages, api_base="http://localhost:11434", stream=True)` |
|Meta LLaMa3 8B | `completion(model='ollama/llama3', messages, api_base="http://localhost:11434", stream=False)` |
| Meta LLaMa3 70B | `completion(model='ollama/llama3:70b', messages, api_base="http://localhost:11434", stream=False)` |
| Orca Mini            | `completion(model='ollama/orca-mini', messages, api_base="http://localhost:11434", stream=True)` |
| Vicuna               | `completion(model='ollama/vicuna', messages, api_base="http://localhost:11434", stream=True)` |
| Nous-Hermes          | `completion(model='ollama/nous-hermes', messages, api_base="http://localhost:11434", stream=True)` |
| Nous-Hermes 13B     | `completion(model='ollama/nous-hermes:13b', messages, api_base="http://localhost:11434", stream=True)` | 
| Wizard Vicuna Uncensored | `completion(model='ollama/wizard-vicuna', messages, api_base="http://localhost:11434", stream=True)` |


### JSON Schema support 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

response = completion(
    model="ollama_chat/deepseek-r1", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}},
)
print(response)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml 

```yaml
model_list:
  - model_name: "deepseek-r1"             
    litellm_params:
      model: "ollama_chat/deepseek-r1"
      api_base: "http://localhost:11434"
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml

# RUNNING ON http://0.0.0.0:4000
```

3. Test it! 

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key="anything", # 👈 PROXY KEY (can be anything, if master_key not set)
    base_url="http://0.0.0.0:4000" # 👈 PROXY BASE URL
)

class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

completion = client.beta.chat.completions.parse(
    model="deepseek-r1",
    messages=[
        {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
        {"role": "user", "content": "how can I solve 8x + 7 = -23"}
    ],
    response_format=MathReasoning,
)

math_reasoning = completion.choices[0].message.parsed
```
</TabItem>
</Tabs>

## Ollama Vision Models
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
|  llava  | `completion('ollama/llava', messages)` |

#### Using Ollama Vision Models

Call `ollama/llava` in the same input/output format as OpenAI [`gpt-4-vision`](https://docs.litellm.ai/docs/providers/openai#openai-vision-models)

LiteLLM Supports the following image types passed in `url`
- Base64 encoded svgs

**Example Request**
```python
import litellm

response = litellm.completion(
  model = "ollama/llava",
  messages=[
      {
          "role": "user",
          "content": [
                          {
                              "type": "text",
                              "text": "Whats in this image?"
                          },
                          {
                              "type": "image_url",
                              "image_url": {
                              "url": "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"
                              }
                          }
                      ]
      }
  ],
)
print(response)
```



## LiteLLM/Ollama Docker Image 

For Ollama LiteLLM Provides a Docker Image for an OpenAI API compatible server for local LLMs - llama2, mistral, codellama


[![Chat on WhatsApp](https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square)](https://wa.link/huol9n) [![Chat on Discord](https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square)](https://discord.gg/wuPM9dRgDw) 
### An OpenAI API compatible server for local LLMs - llama2, mistral, codellama

### Quick Start:
Docker Hub: 
For ARM Processors: https://hub.docker.com/repository/docker/litellm/ollama/general
For Intel/AMD Processors: to be added
```shell
docker pull litellm/ollama
```

```shell
docker run --name ollama litellm/ollama
```

#### Test the server container
On the docker container run the `test.py` file using `python3 test.py`


### Making a request to this server
```python
import openai

api_base = f"http://0.0.0.0:4000" # base url for server

openai.api_base = api_base
openai.api_key = "temp-key"
print(openai.api_base)


print(f'LiteLLM: response from proxy with streaming')
response = openai.chat.completions.create(
    model="ollama/llama2", 
    messages = [
        {
            "role": "user",
            "content": "this is a test request, acknowledge that you got it"
        }
    ],
    stream=True
)

for chunk in response:
    print(f'LiteLLM: streaming response from proxy {chunk}')
```

### Responses from this server 
```json
{
  "object": "chat.completion",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": " Hello! I acknowledge receipt of your test request. Please let me know if there's anything else I can assist you with.",
        "role": "assistant",
        "logprobs": null
      }
    }
  ],
  "id": "chatcmpl-403d5a85-2631-4233-92cb-01e6dffc3c39",
  "created": 1696992706.619709,
  "model": "ollama/llama2",
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 25,
    "total_tokens": 43
  }
}
```

## Calling Docker Container (host.docker.internal)

[Follow these instructions](https://github.com/BerriAI/litellm/issues/1517#issuecomment-1922022209/)
