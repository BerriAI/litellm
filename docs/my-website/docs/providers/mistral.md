import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Mistral AI API
https://docs.mistral.ai/api/

## API Key
```python
# env variable
os.environ['MISTRAL_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['MISTRAL_API_KEY'] = ""
response = completion(
    model="mistral/mistral-tiny", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['MISTRAL_API_KEY'] = ""
response = completion(
    model="mistral/mistral-tiny", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```



## Usage with LiteLLM Proxy 

### 1. Set Mistral Models on config.yaml

```yaml
model_list:
  - model_name: mistral-small-latest
    litellm_params:
      model: mistral/mistral-small-latest
      api_key: "os.environ/MISTRAL_API_KEY" # ensure you have `MISTRAL_API_KEY` in your .env
```

### 2. Start Proxy 

```
litellm --config config.yaml
```

### 3. Test it


<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "mistral-small-latest",
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

response = client.chat.completions.create(model="mistral-small-latest", messages = [
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
    model = "mistral-small-latest",
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

## Supported Models

:::info
All models listed here https://docs.mistral.ai/platform/endpoints are supported. We actively maintain the list of models, pricing, token window, etc. [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

:::


| Model Name     | Function Call                                                |
|----------------|--------------------------------------------------------------|
| Mistral Small  | `completion(model="mistral/mistral-small-latest", messages)` |
| Mistral Medium | `completion(model="mistral/mistral-medium-latest", messages)`|
| Mistral Large 2  | `completion(model="mistral/mistral-large-2407", messages)` |
| Mistral Large Latest  | `completion(model="mistral/mistral-large-latest", messages)` |
| Mistral 7B     | `completion(model="mistral/open-mistral-7b", messages)`      |
| Mixtral 8x7B   | `completion(model="mistral/open-mixtral-8x7b", messages)`    |
| Mixtral 8x22B  | `completion(model="mistral/open-mixtral-8x22b", messages)`   |
| Codestral      | `completion(model="mistral/codestral-latest", messages)`     |
| Mistral NeMo      | `completion(model="mistral/open-mistral-nemo", messages)`     |
| Mistral NeMo 2407      | `completion(model="mistral/open-mistral-nemo-2407", messages)`     |
| Codestral Mamba      | `completion(model="mistral/open-codestral-mamba", messages)`     |
| Codestral Mamba    | `completion(model="mistral/codestral-mamba-latest"", messages)`     |

## Function Calling 

```python
from litellm import completion

# set env
os.environ["MISTRAL_API_KEY"] = "your-api-key"

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
    model="mistral/mistral-large-latest",
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

## Sample Usage - Embedding
```python
from litellm import embedding
import os

os.environ['MISTRAL_API_KEY'] = ""
response = embedding(
    model="mistral/mistral-embed",
    input=["good morning from litellm"],
)
print(response)
```


## Supported Models
All models listed here https://docs.mistral.ai/platform/endpoints are supported

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Mistral Embeddings | `embedding(model="mistral/mistral-embed", input)` | 


