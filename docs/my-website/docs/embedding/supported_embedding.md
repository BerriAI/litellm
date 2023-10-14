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

## Cohere Embedding Models
https://docs.cohere.com/reference/embed

### Usage
```python
from litellm import embedding
import os
os.environ['COHERE_API_KEY'] = ""
response = embedding('embed-english-v2.0', input=["good morning from litellm"])
```

| Model Name            | Function Call | Required OS Variables                        |
|-----------------------|--------------------------------------------------------------|-------------------------------------------------|
| embed-english-v2.0    | `embedding('embed-english-v2.0', input=input)`               | `os.environ['COHERE_API_KEY']`                                             |
| embed-english-light-v2.0 | `embedding('embed-english-light-v2.0', input=input)`         | `os.environ['COHERE_API_KEY']`                                             |
| embed-multilingual-v2.0 | `embedding('embed-multilingual-v2.0', input=input)`         | `os.environ['COHERE_API_KEY']`                                             |

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

