import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Groq
https://groq.com/

:::tip

**We support ALL Groq models, just set `model=groq/<any-model-on-groq>` as a prefix when sending litellm requests**

:::

## API Key
```python
# env variable
os.environ['GROQ_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['GROQ_API_KEY'] = ""
response = completion(
    model="groq/llama3-8b-8192", 
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

os.environ['GROQ_API_KEY'] = ""
response = completion(
    model="groq/llama3-8b-8192", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```



## Usage with LiteLLM Proxy 

### 1. Set Groq Models on config.yaml

```yaml
model_list:
  - model_name: groq-llama3-8b-8192 # Model Alias to use for requests
    litellm_params:
      model: groq/llama3-8b-8192
      api_key: "os.environ/GROQ_API_KEY" # ensure you have `GROQ_API_KEY` in your .env
```

### 2. Start Proxy 

```
litellm --config config.yaml
```

### 3. Test it

Make request to litellm proxy

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "groq-llama3-8b-8192",
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

response = client.chat.completions.create(model="groq-llama3-8b-8192", messages = [
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
    model = "groq-llama3-8b-8192",
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



## Supported Models - ALL Groq Models Supported!
We support ALL Groq models, just set `groq/` as a prefix when sending completion requests

| Model Name         | Usage                                           |
|--------------------|---------------------------------------------------------|
| llama-3.1-8b-instant     | `completion(model="groq/llama-3.1-8b-instant", messages)`     | 
| llama-3.1-70b-versatile    | `completion(model="groq/llama-3.1-70b-versatile", messages)`    | 
| llama3-8b-8192     | `completion(model="groq/llama3-8b-8192", messages)`     | 
| llama3-70b-8192    | `completion(model="groq/llama3-70b-8192", messages)`    | 
| llama2-70b-4096    | `completion(model="groq/llama2-70b-4096", messages)`    | 
| mixtral-8x7b-32768 | `completion(model="groq/mixtral-8x7b-32768", messages)` |
| gemma-7b-it        | `completion(model="groq/gemma-7b-it", messages)`        |
| moonshotai/kimi-k2-instruct | `completion(model="groq/moonshotai/kimi-k2-instruct", messages)` |
| qwen3-32b       | `completion(model="groq/qwen/qwen3-32b", messages)`       |

## Groq - Tool / Function Calling Example

```python
# Example dummy function hard coded to return the current weather
import json
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})




# Step 1: send the conversation and available functions to the model
messages = [
    {
        "role": "system",
        "content": "You are a function calling LLM that uses the data extracted from get_current_weather to answer questions about the weather in San Francisco.",
    },
    {
        "role": "user",
        "content": "What's the weather like in San Francisco?",
    },
]
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
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]
response = litellm.completion(
    model="groq/llama3-8b-8192",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
print("Response\n", response)
response_message = response.choices[0].message
tool_calls = response_message.tool_calls


# Step 2: check if the model wanted to call a function
if tool_calls:
    # Step 3: call the function
    # Note: the JSON response may not always be valid; be sure to handle errors
    available_functions = {
        "get_current_weather": get_current_weather,
    }
    messages.append(
        response_message
    )  # extend conversation with assistant's reply
    print("Response message\n", response_message)
    # Step 4: send the info for each function call and function response to the model
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
    print(f"messages: {messages}")
    second_response = litellm.completion(
        model="groq/llama3-8b-8192", messages=messages
    )  # get a new response from the model where it can see the function response
    print("second response\n", second_response)
```

## Groq - Vision Example    

Select Groq models support vision. Check out their [model list](https://console.groq.com/docs/vision) for more details.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

import os 
from litellm import completion

os.environ["GROQ_API_KEY"] = "your-api-key"

# openai call
response = completion(
    model = "groq/llama-3.2-11b-vision-preview", 
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
                                "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
                                }
                            }
                        ]
        }
    ],
)

```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add Groq models to config.yaml   

```yaml
model_list:
  - model_name: groq-llama3-8b-8192 # Model Alias to use for requests
    litellm_params:
      model: groq/llama3-8b-8192
      api_key: "os.environ/GROQ_API_KEY" # ensure you have `GROQ_API_KEY` in your .env
```

2. Start Proxy

```bash
litellm --config config.yaml
```

3. Test it

```python
import os 
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234", # your litellm proxy api key
)

response = client.chat.completions.create(
    model = "gpt-4-vision-preview",  # use model="llava-hf" to test your custom OpenAI endpoint
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
                                "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
                                }
                            }
                        ]
        }
    ],
)

```
</TabItem>
</Tabs>

## Speech to Text - Whisper

```python
os.environ["GROQ_API_KEY"] = ""
audio_file = open("/path/to/audio.mp3", "rb")

transcript = litellm.transcription(
    model="groq/whisper-large-v3",
    file=audio_file,
    prompt="Specify context or spelling",
    temperature=0,
    response_format="json"
)

print("response=", transcript)
```

