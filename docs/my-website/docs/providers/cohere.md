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

### Setting - Input Type for v3 models
v3 Models have a required parameter: `input_type`, it can be one of the following four values:

- `input_type="search_document"`: (default) Use this for texts (documents) you want to store in your vector database
- `input_type="search_query"`: Use this for search queries to find the most relevant documents in your vector database
- `input_type="classification"`: Use this if you use the embeddings as an input for a classification system
- `input_type="clustering"`: Use this if you use the embeddings for text clustering

https://txt.cohere.com/introducing-embed-v3/
```python
from litellm import embedding
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = embedding(
    model="embed-english-v3.0", 
    input=["good morning from litellm", "this is another item"], 
    input_type="search_document" 
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

