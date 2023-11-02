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

## Embedding

```python
from litellm import embedding
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = embedding(
    model="embed-english-v3.0", 
    input=["good morning from litellm", "this is another item"], 
)
```

### Supported Embedding Models
| Model Name               | Function Call                                                |
|--------------------------|--------------------------------------------------------------|
| embed-english-v3.0       | `embedding(model="embed-english-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-english-light-v3.0 | `embedding(model="embed-english-light-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-multilingual-v3.0  | `embedding(model="embed-multilingual-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-multilingual-light-v3.0 | `embedding(model="embed-multilingual-light-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-english-v2.0       | `embedding(model="embed-english-v2.0", input=["good morning from litellm", "this is another item"])` |
| embed-english-light-v2.0 | `embedding(model="embed-english-light-v2.0", input=["good morning from litellm", "this is another item"])` |
| embed-multilingual-v2.0  | `embedding(model="embed-multilingual-v2.0", input=["good morning from litellm", "this is another item"])` |

