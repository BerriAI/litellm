# Embedding Models

## Usage
```python
from litellm import embedding
import os
os.environ['OPENAI_API_KEY'] = ""
response = embedding('text-embedding-ada-002', input=["good morning from litellm"])
```

## OpenAI Embedding Models

| Model Name           | Function Call                               | Required OS Variables                |
|----------------------|---------------------------------------------|--------------------------------------|
| text-embedding-ada-002 | `embedding('text-embedding-ada-002', input)` | `os.environ['OPENAI_API_KEY']`       |

## Azure OpenAI Embedding Models

| Model Name           | Function Call                               | Required OS Variables                |
|----------------------|---------------------------------------------|--------------------------------------|
| text-embedding-ada-002 | `embedding('embedding-model-deployment', input=input, custom_llm_provider="azure")` | `os.environ['AZURE_API_KEY']`,`os.environ['AZURE_API_BASE']`,`os.environ['AZURE_API_VERSION']` |