# OpenAI
LiteLLM supports OpenAI Chat + Text completion and embedding calls.

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

### Optional Keys - OpenAI Organization, OpenAI API Base

```python
import os 
os.environ["OPENAI_ORGANIZATION"] = "your-org-id"       # OPTIONAL
os.environ["OPENAI_API_BASE"] = "openaiai-api-base"     # OPTIONAL
```

### OpenAI Chat Completion Models

| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
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

## OpenAI Text Completion Models / Instruct Models

| Model Name          | Function Call                                      |
|---------------------|----------------------------------------------------|
| gpt-3.5-turbo-instruct | `response = completion(model="gpt-3.5-turbo-instruct", messages=messages)` |
| text-davinci-003    | `response = completion(model="text-davinci-003", messages=messages)` |
| ada-001             | `response = completion(model="ada-001", messages=messages)` |
| curie-001           | `response = completion(model="curie-001", messages=messages)` |
| babbage-001         | `response = completion(model="babbage-001", messages=messages)` |
| babbage-002         | `response = completion(model="babbage-002", messages=messages)` |
| davinci-002         | `response = completion(model="davinci-002", messages=messages)` |

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