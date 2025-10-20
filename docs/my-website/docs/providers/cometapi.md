# CometAPI
LiteLLM supports all AI models from [CometAPI](https://www.cometapi.com/). CometAPI provides access to 500+ AI models through a unified API interface, including cutting-edge models like GPT-5, Claude Opus 4.1, and various other state-of-the-art language models.

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_CometAPI.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Authentication

To use CometAPI models, you need to obtain an API key from [CometAPI Token Console](https://api.cometapi.com/console/token). CometAPI offers free tokens for new users - you can get your free API key instantly by registering.

## Usage

Set your CometAPI key as an environment variable and use the completion function:

```python
import os
from litellm import completion

# Set API key
os.environ["COMETAPI_KEY"] = "your_comet_api_key_here"

# Define messages
messages = [{"content": "Hello, how are you?", "role": "user"}]

# Method 1: Using environment variable (recommended)
response = completion(
    model="cometapi/gpt-5", 
    messages=messages
)

print(response.choices[0].message.content)
```

### Alternative Usage - Explicit API Key

You can also pass the API key explicitly:

```python
import os
from litellm import completion

# Define messages
messages = [{"content": "Hello, how are you?", "role": "user"}]

# Method 2: Explicitly passing API key
response = completion(
    model="cometapi/gpt-4o", 
    messages=messages, 
    api_key="your_comet_api_key_here"
)

print(response.choices[0].message.content)
```

## Usage - Streaming

Just set `stream=True` when calling completion:

```python
import os
from litellm import completion

os.environ["COMETAPI_KEY"] = "your_comet_api_key_here"

messages = [{"content": "Hello, how are you?", "role": "user"}]

response = completion(
    model="cometapi/gpt-5",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Usage - Async Streaming

For async streaming, use `acompletion`:

```python
from litellm import acompletion
import asyncio, os, traceback

async def completion_call():
    try:
        os.environ["COMETAPI_KEY"] = "your_comet_api_key_here"
        
        print("test acompletion + streaming")
        response = await acompletion(
            model="cometapi/chatgpt-4o-latest", 
            messages=[{"content": "Hello, how are you?", "role": "user"}], 
            stream=True
        )
        print(f"response: {response}")
        async for chunk in response:
            print(chunk)
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass

# Run the async function
await completion_call()
```

## CometAPI Models

CometAPI offers access to 500+ AI models through a unified API. Some popular models include:

| Model Name | Function Call |
|------------|---------------|
| cometapi/gpt-5 | `completion('cometapi/gpt-5', messages)` |
| cometapi/gpt-5-mini | `completion('cometapi/gpt-5-mini', messages)` |
| cometapi/gpt-5-nano | `completion('cometapi/gpt-5-nano', messages)` |
| cometapi/gpt-oss-20b | `completion('cometapi/gpt-oss-20b', messages)` |
| cometapi/gpt-oss-120b | `completion('cometapi/gpt-oss-120b', messages)` |
| cometapi/chatgpt-4o-latest | `completion('cometapi/chatgpt-4o-latest', messages)` |

For a complete list of available models, visit the [CometAPI Models page](https://www.cometapi.com/model/).

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `COMETAPI_KEY` | Your CometAPI API key | Yes |

## Error Handling

```python
import os
from litellm import completion

try:
    os.environ["COMETAPI_KEY"] = "your_comet_api_key_here"
    
    messages = [{"content": "Hello, how are you?", "role": "user"}]
    
    response = completion(
        model="cometapi/gpt-5",
        messages=messages
    )
    
    print(response.choices[0].message.content)
    
except Exception as e:
    print(f"Error: {e}")
```
