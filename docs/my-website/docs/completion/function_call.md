# Function Calling 
LiteLLM only supports: OpenAI gpt-4-0613 and gpt-3.5-turbo-0613 for function calling 
## Quick Start 
```python
import os, litellm
from litellm import completion

os.environ['OPENAI_API_KEY'] = ""

messages = [
    {"role": "user", "content": "What is the weather like in Boston?"}
]

def get_current_weather(location):
  if location == "Boston, MA":
    return "The weather is 12F"

functions = [
    {
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
  ]

response = completion(model="gpt-3.5-turbo-0613", messages=messages, functions=functions)
print(response)
```

## Using - litellm.utils.function_to_dict
`function_to_dict` allows you to pass a function docstring and produce a dictionary usable for OpenAI function calling

### Usage
Define your function, use `litellm.utils.function_to_dict` to convert your function to a dictionary usable for OpenAI 

```python
def get_current_weather(location: str, unit: str):
        """Get the current weather in a given location

        Parameters
        ----------
        location : str
            The city and state, e.g. San Francisco, CA
        unit : {'celsius', 'fahrenheit'}
            Temperature unit

        Returns
        -------
        str
            a sentence indicating the weather
        """
        if location == "Boston, MA":
            return "The weather is 12F"
    function_json = litellm.utils.function_to_dict(get_current_weather)
    print(function_json)
```

#### Output
```json
{
    'name': 'get_current_weather', 
    'description': 'Get the current weather in a given location', 
    'parameters': {
        'type': 'object', 
        'properties': {
            'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}, 
            'unit': {'type': 'string', 'description': 'Temperature unit', 'enum': "['fahrenheit', 'celsius']"}
        }, 
        'required': ['location', 'unit']
    }
}
```

### Using function_to_dict with Function calling
```python
import os, litellm
from litellm import completion

os.environ['OPENAI_API_KEY'] = ""

messages = [
    {"role": "user", "content": "What is the weather like in Boston?"}
]

def get_current_weather(location):
  if location == "Boston, MA":
    return "The weather is 12F"

functions = litellm.utils.function_to_dict(get_current_weather)

response = completion(model="gpt-3.5-turbo-0613", messages=messages, functions=functions)
print(response)
```