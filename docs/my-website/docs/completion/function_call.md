# Function Calling 
LiteLLM only supports: OpenAI gpt-4-0613 and gpt-3.5-turbo-0613 for function calling 
## Quick Start 
This is exactly how OpenAI supports function calling for gpt-4-0613 and gpt-3.5-turbo-0613
```python
import os, litellm
from litellm import completion

os.environ['OPENAI_API_KEY'] = ""

messages = [
    {"role": "user", "content": "What is the weather like in Boston?"}
]

# python function that will get executed
def get_current_weather(location):
  if location == "Boston, MA":
    return "The weather is 12F"

# JSON Schema to pass to OpenAI
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

## litellm.function_to_dict - Convert Functions to dictionary for OpenAI function calling
`function_to_dict` allows you to pass a function docstring and produce a dictionary usable for OpenAI function calling

### Using `function_to_dict`
1. Define your function `get_current_weather`
2. Add a docstring to your function `get_current_weather`
3. Pass the function to `litellm.utils.function_to_dict` to get the dictionary for OpenAI function calling

```python
# function with docstring
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

# use litellm.utils.function_to_dict to convert function to dict
function_json = litellm.utils.function_to_dict(get_current_weather)
print(function_json)
```

#### Output from function_to_dict
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

def get_current_weather(location: str, unit: str):
    """Get the current weather in a given location

    Parameters
    ----------
    location : str
        The city and state, e.g. San Francisco, CA
    unit : str {'celsius', 'fahrenheit'}
        Temperature unit

    Returns
    -------
    str
        a sentence indicating the weather
    """
    if location == "Boston, MA":
        return "The weather is 12F"

functions = [litellm.utils.function_to_dict(get_current_weather)]

response = completion(model="gpt-3.5-turbo-0613", messages=messages, functions=functions)
print(response)
```

## Function calling for Non-OpenAI LLMs
**For Non OpenAI LLMs - LiteLLM raises an exception if you try using it for function calling**

### Adding Function to prompt
For Non OpenAI LLMs LiteLLM allows you to add the function to the prompt set: `litellm.add_function_to_prompt = True`

#### Usage
```python
import os, litellm
from litellm import completion

# IMPORTANT - Set this to TRUE to add the function to the prompt for Non OpenAI LLMs
litellm.add_function_to_prompt = True # set add_function_to_prompt for Non OpenAI LLMs

os.environ['ANTHROPIC_API_KEY'] = ""

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

response = completion(model="claude-2", messages=messages, functions=functions)
print(response)
```

