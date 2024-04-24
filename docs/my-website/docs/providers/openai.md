import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI
LiteLLM supports OpenAI Chat + Embedding calls.

### Required API Keys

```python
import os 
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

### Usage
```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"

# openai call
response = completion(
    model = "gpt-3.5-turbo", 
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

### Usage - LiteLLM Proxy Server

Here's how to call OpenAI models with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export OPENAI_API_KEY=""
```

### 2. Start the proxy 

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo                          # The `openai/` prefix will call openai.chat.completions.create
      api_key: os.environ/OPENAI_API_KEY
  - model_name: gpt-3.5-turbo-instruct
    litellm_params:
      model: text-completion-openai/gpt-3.5-turbo-instruct # The `text-completion-openai/` prefix will call openai.completions.create
      api_key: os.environ/OPENAI_API_KEY
```
</TabItem>
<TabItem value="config-*" label="config.yaml - proxy all OpenAI models">

Use this to add all openai models with one API Key. **WARNING: This will not do any load balancing**
This means requests to `gpt-4`, `gpt-3.5-turbo` , `gpt-4-turbo-preview` will all go through this route 

```yaml
model_list:
  - model_name: "*"             # all requests where model not in your config go to this deployment
    litellm_params:
      model: openai/*           # set `openai/` to use the openai route
      api_key: os.environ/OPENAI_API_KEY
```
</TabItem>
<TabItem value="cli" label="CLI">

```bash
$ litellm --model gpt-3.5-turbo

# Server running on http://0.0.0.0:4000
```
</TabItem>

</Tabs>

### 3. Test it


<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)

```
</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000", # set openai_api_base to the LiteLLM Proxy
    model = "gpt-3.5-turbo",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```
</TabItem>
</Tabs>


### Optional Keys - OpenAI Organization, OpenAI API Base

```python
import os 
os.environ["OPENAI_ORGANIZATION"] = "your-org-id"       # OPTIONAL
os.environ["OPENAI_API_BASE"] = "openaiai-api-base"     # OPTIONAL
```

### OpenAI Chat Completion Models

| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
| gpt-4-turbo   | `response = completion(model="gpt-4-turbo", messages=messages)` |
| gpt-4-turbo-preview   | `response = completion(model="gpt-4-0125-preview", messages=messages)` |
| gpt-4-0125-preview    | `response = completion(model="gpt-4-0125-preview", messages=messages)` |
| gpt-4-1106-preview    | `response = completion(model="gpt-4-1106-preview", messages=messages)` |
| gpt-3.5-turbo-1106    | `response = completion(model="gpt-3.5-turbo-1106", messages=messages)` |
| gpt-3.5-turbo         | `response = completion(model="gpt-3.5-turbo", messages=messages)` |
| gpt-3.5-turbo-0301    | `response = completion(model="gpt-3.5-turbo-0301", messages=messages)` |
| gpt-3.5-turbo-0613    | `response = completion(model="gpt-3.5-turbo-0613", messages=messages)` |
| gpt-3.5-turbo-16k     | `response = completion(model="gpt-3.5-turbo-16k", messages=messages)` |
| gpt-3.5-turbo-16k-0613| `response = completion(model="gpt-3.5-turbo-16k-0613", messages=messages)` |
| gpt-4                 | `response = completion(model="gpt-4", messages=messages)` |
| gpt-4-0314            | `response = completion(model="gpt-4-0314", messages=messages)` |
| gpt-4-0613            | `response = completion(model="gpt-4-0613", messages=messages)` |
| gpt-4-32k             | `response = completion(model="gpt-4-32k", messages=messages)` |
| gpt-4-32k-0314        | `response = completion(model="gpt-4-32k-0314", messages=messages)` |
| gpt-4-32k-0613        | `response = completion(model="gpt-4-32k-0613", messages=messages)` |


These also support the `OPENAI_API_BASE` environment variable, which can be used to specify a custom API endpoint.

## OpenAI Vision Models 
| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
| gpt-4-turbo    | `response = completion(model="gpt-4-turbo", messages=messages)` |
| gpt-4-vision-preview    | `response = completion(model="gpt-4-vision-preview", messages=messages)` |

#### Usage
```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"

# openai call
response = completion(
    model = "gpt-4-vision-preview", 
    messages=[
        {
            "role": "user",
            "content": [
                            {
                                "type": "text",
                                "text": "What’s in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                }
                            }
                        ]
        }
    ],
)

```

## Advanced

### Parallel Function calling
See a detailed walthrough of parallel function calling with litellm [here](https://docs.litellm.ai/docs/completion/function_call)
```python
import litellm
import json
# set openai api key
import os
os.environ['OPENAI_API_KEY'] = "" # litellm reads OPENAI_API_KEY from .env and sends the request
# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
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

response = litellm.completion(
    model="gpt-3.5-turbo-1106",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
print("\nLLM Response1:\n", response)
response_message = response.choices[0].message
tool_calls = response.choices[0].message.tool_calls
```

### Setting `extra_headers` for completion calls
```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"

response = completion(
    model = "gpt-3.5-turbo", 
    messages=[{ "content": "Hello, how are you?","role": "user"}],
    extra_headers={"AI-Resource Group": "ishaan-resource"}
)
```

### Setting Organization-ID for completion calls
This can be set in one of the following ways:
- Environment Variable `OPENAI_ORGANIZATION`
- Params to `litellm.completion(model=model, organization="your-organization-id")`
- Set as `litellm.organization="your-organization-id"`
```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_ORGANIZATION"] = "your-org-id" # OPTIONAL

response = completion(
    model = "gpt-3.5-turbo", 
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

### Set `ssl_verify=False`

This is done by setting your own `httpx.Client` 

- For `litellm.completion` set `litellm.client_session=httpx.Client(verify=False)`
- For `litellm.acompletion` set `litellm.aclient_session=AsyncClient.Client(verify=False)`
```python
import litellm, httpx

# for completion
litellm.client_session = httpx.Client(verify=False)
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=messages,
)

# for acompletion
litellm.aclient_session = httpx.AsyncClient(verify=False)
response = litellm.acompletion(
    model="gpt-3.5-turbo",
    messages=messages,
)
```

### Using Helicone Proxy with LiteLLM
```python
import os 
import litellm
from litellm import completion

os.environ["OPENAI_API_KEY"] = ""

# os.environ["OPENAI_API_BASE"] = ""
litellm.api_base = "https://oai.hconeai.com/v1"
litellm.headers = {
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",
    "Helicone-Cache-Enabled": "true",
}

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion("gpt-3.5-turbo", messages)
```

### Using OpenAI Proxy with LiteLLM
```python
import os 
import litellm
from litellm import completion

os.environ["OPENAI_API_KEY"] = ""

# set custom api base to your proxy
# either set .env or litellm.api_base
# os.environ["OPENAI_API_BASE"] = ""
litellm.api_base = "your-openai-proxy-url"


messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion("openai/your-model-name", messages)
```

If you need to set api_base dynamically, just pass it in completions instead - `completions(...,api_base="your-proxy-api-base")`

For more check out [setting API Base/Keys](../set_keys.md)