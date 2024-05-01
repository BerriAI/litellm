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


## Supported Models
All models listed here https://docs.mistral.ai/platform/endpoints are supported. We actively maintain the list of models, pricing, token window, etc. [here](https://github.com/BerriAI/litellm/blob/c1b25538277206b9f00de5254d80d6a83bb19a29/model_prices_and_context_window.json).

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mistral-tiny | `completion(model="mistral/mistral-tiny", messages)` | 
| mistral-small | `completion(model="mistral/mistral-small", messages)` | 
| mistral-medium | `completion(model="mistral/mistral-medium", messages)` | 
| mistral-large-latest | `completion(model="mistral/mistral-large-latest", messages)` | 
| open-mixtral-8x22b | `completion(model="mistral/open-mixtral-8x22b", messages)` | 


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
| mistral-embed | `embedding(model="mistral/mistral-embed", input)` | 


