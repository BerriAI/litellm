# Embedding Models

## OpenAI Embedding Models

| Model Name           | Function Call                               | Required OS Variables                |
|----------------------|---------------------------------------------|--------------------------------------|
| text-embedding-ada-002 | `embedding('text-embedding-ada-002', input)` | `os.environ['OPENAI_API_KEY']`       |

## Azure OpenAI Embedding Models

| Model Name           | Function Call                               | Required OS Variables                |
|----------------------|---------------------------------------------|--------------------------------------|
| text-embedding-ada-002 | `embedding('embedding-model-deployment', input=input, custom_llm_provider="azure")` | `os.environ['AZURE_API_KEY']`,`os.environ['AZURE_API_BASE']`,`os.environ['AZURE_API_VERSION']` |

# Moderation
LiteLLM supports the moderation endpoint for OpenAI

## Usage
```python
import os
from litellm import moderation
os.environ['OPENAI_API_KEY'] = ""
response = moderation(input="i'm ishaan cto of litellm")   
```
