# Embedding Models

## Quick Start
```python
from litellm import embedding
import os
os.environ['OPENAI_API_KEY'] = ""
response = embedding('text-embedding-ada-002', input=["good morning from litellm"])
```

## OpenAI Embedding Models

### Usage
```python
from litellm import embedding
import os
os.environ['OPENAI_API_KEY'] = ""
response = embedding('text-embedding-ada-002', input=["good morning from litellm"])
```

| Model Name           | Function Call                               | Required OS Variables                |
|----------------------|---------------------------------------------|--------------------------------------|
| text-embedding-ada-002 | `embedding('text-embedding-ada-002', input)` | `os.environ['OPENAI_API_KEY']`       |

## Azure OpenAI Embedding Models

### API keys
This can be set as env variables or passed as **params to litellm.embedding()**
```python
import os
os.environ['AZURE_API_KEY'] = 
os.environ['AZURE_API_BASE'] = 
os.environ['AZURE_API_VERSION'] = 
```

### Usage
```python
from litellm import embedding
response = embedding(
    model="azure/<your deployment name>",
    input=["good morning from litellm"],
    api_key=api_key,
    api_base=api_base,
    api_version=api_version,
)
print(response)
```

| Model Name           | Function Call                               |
|----------------------|---------------------------------------------|
| text-embedding-ada-002 | `embedding(model="azure/<your deployment name>", input=input)` |

h/t to [Mikko](https://www.linkedin.com/in/mikkolehtimaki/) for this integration


## Bedrock Embedding

### API keys
This can be set as env variables or passed as **params to litellm.embedding()**
```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = ""  # Access key
os.environ["AWS_SECRET_ACCESS_KEY"] = "" # Secret access key
os.environ["AWS_REGION_NAME"] = "" # us-east-1, us-east-2, us-west-1, us-west-2
```

### Usage
```python
from litellm import embedding
response = embedding(
    model="amazon.titan-embed-text-v1",
    input=["good morning from litellm"],
)
print(response)
```

| Model Name           | Function Call                               |
|----------------------|---------------------------------------------|
| Titan Embeddings - G1 | `embedding(model="amazon.titan-embed-text-v1", input=input)` |


## Cohere Embedding Models
https://docs.cohere.com/reference/embed

### Usage
```python
from litellm import embedding
os.environ["COHERE_API_KEY"] = "cohere key"

# cohere call
response = embedding(
    model="embed-english-v3.0", 
    input=["good morning from litellm", "this is another item"], 
    input_type="search_document" # optional param for v3 llms
)
```
| Model Name               | Function Call                                                |
|--------------------------|--------------------------------------------------------------|
| embed-english-v3.0       | `embedding(model="embed-english-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-english-light-v3.0 | `embedding(model="embed-english-light-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-multilingual-v3.0  | `embedding(model="embed-multilingual-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-multilingual-light-v3.0 | `embedding(model="embed-multilingual-light-v3.0", input=["good morning from litellm", "this is another item"])` |
| embed-english-v2.0       | `embedding(model="embed-english-v2.0", input=["good morning from litellm", "this is another item"])` |
| embed-english-light-v2.0 | `embedding(model="embed-english-light-v2.0", input=["good morning from litellm", "this is another item"])` |
| embed-multilingual-v2.0  | `embedding(model="embed-multilingual-v2.0", input=["good morning from litellm", "this is another item"])` |

## HuggingFace Embedding Models
LiteLLM supports all Feature-Extraction Embedding models: https://huggingface.co/models?pipeline_tag=feature-extraction

### Usage
```python
from litellm import embedding
import os
os.environ['HUGGINGFACE_API_KEY'] = ""
response = embedding(
    model='huggingface/microsoft/codebert-base', 
    input=["good morning from litellm"]
)
```

| Model Name            | Function Call | Required OS Variables                        |
|-----------------------|--------------------------------------------------------------|-------------------------------------------------|
| microsoft/codebert-base    | `embedding('huggingface/microsoft/codebert-base', input=input)`               | `os.environ['HUGGINGFACE_API_KEY']`                                             |
| BAAI/bge-large-zh | `embedding('huggingface/BAAI/bge-large-zh', input=input)`         | `os.environ['HUGGINGFACE_API_KEY']`                                             |
| any-hf-embedding-model | `embedding('huggingface/hf-embedding-model', input=input)`         | `os.environ['HUGGINGFACE_API_KEY']`                                             |

