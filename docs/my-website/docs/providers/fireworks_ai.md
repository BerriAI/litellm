import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Fireworks AI
https://fireworks.ai/

:::info
**We support ALL Fireworks AI models, just set `fireworks_ai/` as a prefix when sending completion requests**
:::

## API Key
```python
# env variable
os.environ['FIREWORKS_AI_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['FIREWORKS_AI_API_KEY'] = ""
response = completion(
    model="fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct", 
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

os.environ['FIREWORKS_AI_API_KEY'] = ""
response = completion(
    model="fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```


## Usage with LiteLLM Proxy 

### 1. Set Fireworks AI Models on config.yaml

```yaml
model_list:
  - model_name: fireworks-llama-v3-70b-instruct
    litellm_params:
      model: fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct
      api_key: "os.environ/FIREWORKS_AI_API_KEY"
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
      "model": "fireworks-llama-v3-70b-instruct",
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
response = client.chat.completions.create(model="fireworks-llama-v3-70b-instruct", messages = [
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
    model = "fireworks-llama-v3-70b-instruct",
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

## Supported Models - ALL Fireworks AI Models Supported!

:::info
We support ALL Fireworks AI models, just set `fireworks_ai/` as a prefix when sending completion requests
:::

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| llama-v3p2-1b-instruct | `completion(model="fireworks_ai/llama-v3p2-1b-instruct", messages)` |
| llama-v3p2-3b-instruct | `completion(model="fireworks_ai/llama-v3p2-3b-instruct", messages)` |
| llama-v3p2-11b-vision-instruct | `completion(model="fireworks_ai/llama-v3p2-11b-vision-instruct", messages)` |
| llama-v3p2-90b-vision-instruct | `completion(model="fireworks_ai/llama-v3p2-90b-vision-instruct", messages)` |
| mixtral-8x7b-instruct | `completion(model="fireworks_ai/mixtral-8x7b-instruct", messages)` | 
| firefunction-v1 | `completion(model="fireworks_ai/firefunction-v1", messages)` |
| llama-v2-70b-chat | `completion(model="fireworks_ai/llama-v2-70b-chat", messages)` |  

## Supported Embedding Models

:::info
We support ALL Fireworks AI models, just set `fireworks_ai/` as a prefix when sending embedding requests
:::

| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
| fireworks_ai/nomic-ai/nomic-embed-text-v1.5 | `response = litellm.embedding(model="fireworks_ai/nomic-ai/nomic-embed-text-v1.5", input=input_text)` |
| fireworks_ai/nomic-ai/nomic-embed-text-v1 | `response = litellm.embedding(model="fireworks_ai/nomic-ai/nomic-embed-text-v1", input=input_text)` |
| fireworks_ai/WhereIsAI/UAE-Large-V1 | `response = litellm.embedding(model="fireworks_ai/WhereIsAI/UAE-Large-V1", input=input_text)` |
| fireworks_ai/thenlper/gte-large | `response = litellm.embedding(model="fireworks_ai/thenlper/gte-large", input=input_text)` |
| fireworks_ai/thenlper/gte-base | `response = litellm.embedding(model="fireworks_ai/thenlper/gte-base", input=input_text)` |