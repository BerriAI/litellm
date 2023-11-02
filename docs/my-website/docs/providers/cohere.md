# Cohere

## API KEYS

```python
import os 
os.environ["COHERE_API_KEY"] = ""
```

## Usage

```python
from litellm import completion

## set ENV variables
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = completion(
    model="command-nightly", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## Usage - Streaming

```python
from litellm import completion

## set ENV variables
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = completion(
    model="command-nightly", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=True
)

for chunk in response:
    print(chunk)
```

LiteLLM supports 'command', 'command-light', 'command-medium', 'command-medium-beta', 'command-xlarge-beta', 'command-nightly' models from [Cohere](https://cohere.com/). 