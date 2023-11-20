# Function Calling 
Function calling is supported with the following models on OpenAI, Azure OpenAI

- gpt-4
- gpt-4-1106-preview
- gpt-4-0613
- gpt-3.5-turbo
- gpt-3.5-turbo-1106
- gpt-3.5-turbo-0613
- Non OpenAI LLMs (litellm adds the function call to the prompt for these llms)

In addition, parallel function calls is supported on the following models:
- gpt-4-1106-preview
- gpt-3.5-turbo-1106

## Parallel Function calling
Parallel function calling is the model's ability to perform multiple function calls together, allowing the effects and results of these function calls to be resolved in parallel

## Quick Start - gpt-3.5-turbo-1106
<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/Parallel_function_calling.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

In this example we define a single function `get_current_weather`. 

- Step 1: Send the model the `get_current_weather` with the user question
- Step 2: Parse the output from the model response - Execute the `get_current_weather` with the model provided args
- Step 3: Send the model the output from running the `get_current_weather` function


### Full Code - Parallel function calling with `gpt-3.5-turbo-1106`

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


def test_parallel_function_call():
    try:
        # Step 1: send the conversation and available functions to the model
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
        print("\nFirst LLM Response:\n", response)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        print("\nLength of tool calls", len(tool_calls))

        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "get_current_weather": get_current_weather,
            }  # only one function in this example, but you can have multiple
            messages.append(response_message)  # extend conversation with assistant's reply

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
            second_response = litellm.completion(
                model="gpt-3.5-turbo-1106",
                messages=messages,
            )  # get a new response from the model where it can see the function response
            print("\nSecond LLM response:\n", second_response)
            return second_response
    except Exception as e:
      print(f"Error occurred: {e}")

test_parallel_function_call()
```

### Explanation - Parallel function calling
Below is an explanation of what is happening in the code snippet above for Parallel function calling with `gpt-3.5-turbo-1106`
### Step1: litellm.completion() with `tools` set to `get_current_weather`
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

##### Expected output
In the output you can see the model calls the function multiple times - for San Francisco, Tokyo, Paris
```json
ModelResponse(
  id='chatcmpl-8MHBKZ9t6bXuhBvUMzoKsfmmlv7xq', 
  choices=[
    Choices(finish_reason='tool_calls', 
    index=0, 
    message=Message(content=None, role='assistant', 
      tool_calls=[
        ChatCompletionMessageToolCall(id='call_DN6IiLULWZw7sobV6puCji1O', function=Function(arguments='{"location": "San Francisco", "unit": "celsius"}', name='get_current_weather'), type='function'), 

        ChatCompletionMessageToolCall(id='call_ERm1JfYO9AFo2oEWRmWUd40c', function=Function(arguments='{"location": "Tokyo", "unit": "celsius"}', name='get_current_weather'), type='function'), 
        
        ChatCompletionMessageToolCall(id='call_2lvUVB1y4wKunSxTenR0zClP', function=Function(arguments='{"location": "Paris", "unit": "celsius"}', name='get_current_weather'), type='function')
        ]))
    ], 
    created=1700319953, 
    model='gpt-3.5-turbo-1106', 
    object='chat.completion', 
    system_fingerprint='fp_eeff13170a',
    usage={'completion_tokens': 77, 'prompt_tokens': 88, 'total_tokens': 165}, 
    _response_ms=1177.372
)
```

### Step 2 -  Parse the Model Response and Execute Functions
After sending the initial request, parse the model response to identify the function calls it wants to make. In this example, we expect three tool calls, each corresponding to a location (San Francisco, Tokyo, and Paris). 

```python
# Check if the model wants to call a function
if tool_calls:
    # Execute the functions and prepare responses
    available_functions = {
        "get_current_weather": get_current_weather,
    }

    messages.append(response_message)  # Extend conversation with assistant's reply

    for tool_call in tool_calls:
      print(f"\nExecuting tool call\n{tool_call}")
      function_name = tool_call.function.name
      function_to_call = available_functions[function_name]
      function_args = json.loads(tool_call.function.arguments)
      # calling the get_current_weather() function
      function_response = function_to_call(
          location=function_args.get("location"),
          unit=function_args.get("unit"),
      )
      print(f"Result from tool call\n{function_response}\n")

      # Extend conversation with function response
      messages.append(
          {
              "tool_call_id": tool_call.id,
              "role": "tool",
              "name": function_name,
              "content": function_response,
          }
      )

```

### Step 3 - Second litellm.completion() call 
Once the functions are executed, send the model the information for each function call and its response. This allows the model to generate a new response considering the effects of the function calls.
```python
second_response = litellm.completion(
    model="gpt-3.5-turbo-1106",
    messages=messages,
)
print("Second Response\n", second_response)
```

#### Expected output
```json
ModelResponse(
  id='chatcmpl-8MHBLh1ldADBP71OrifKap6YfAd4w', 
  choices=[
    Choices(finish_reason='stop', index=0, 
    message=Message(content="The current weather in San Francisco is 72°F, in Tokyo it's 10°C, and in Paris it's 22°C.", role='assistant'))
  ], 
  created=1700319955, 
  model='gpt-3.5-turbo-1106', 
  object='chat.completion', 
  system_fingerprint='fp_eeff13170a', 
  usage={'completion_tokens': 28, 'prompt_tokens': 169, 'total_tokens': 197}, 
  _response_ms=1032.431
)
```

## Parallel Function Calling - Azure OpenAI
```python
# set Azure env variables
import os
os.environ['AZURE_API_KEY'] = "" # litellm reads AZURE_API_KEY from .env and sends the request
os.environ['AZURE_API_BASE'] = "https://openai-gpt-4-test-v-1.openai.azure.com/"
os.environ['AZURE_API_VERSION'] = "2023-07-01-preview"

import litellm
import json
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

## Step 1: send the conversation and available functions to the model
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
    model="azure/chatgpt-functioncalling", # model = azure/<your-azure-deployment-name>
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)
print("\nLLM Response1:\n", response)
response_message = response.choices[0].message
tool_calls = response.choices[0].message.tool_calls
print("\nTool Choice:\n", tool_calls)

## Step 2 - Parse the Model Response and Execute Functions
# Check if the model wants to call a function
if tool_calls:
    # Execute the functions and prepare responses
    available_functions = {
        "get_current_weather": get_current_weather,
    }

    messages.append(response_message)  # Extend conversation with assistant's reply

    for tool_call in tool_calls:
      print(f"\nExecuting tool call\n{tool_call}")
      function_name = tool_call.function.name
      function_to_call = available_functions[function_name]
      function_args = json.loads(tool_call.function.arguments)
      # calling the get_current_weather() function
      function_response = function_to_call(
          location=function_args.get("location"),
          unit=function_args.get("unit"),
      )
      print(f"Result from tool call\n{function_response}\n")

      # Extend conversation with function response
      messages.append(
          {
              "tool_call_id": tool_call.id,
              "role": "tool",
              "name": function_name,
              "content": function_response,
          }
      )

## Step 3 - Second litellm.completion() call
second_response = litellm.completion(
    model="azure/chatgpt-functioncalling",
    messages=messages,
)
print("Second Response\n", second_response)
print("Second Response Message\n", second_response.choices[0].message.content)

```

## Deprecated - Function Calling with `completion(functions=functions)`
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

