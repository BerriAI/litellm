# Jina AI
https://jina.ai/embeddings/

## API Key
```python
# env variable
os.environ['JINA_AI_API_KEY']
```

## Sample Usage - Embedding
```python
from litellm import embedding
import os

os.environ['JINA_AI_API_KEY'] = ""
response = embedding(
    model="jina_ai/jina-embeddings-v3",
    input=["good morning from litellm"],
)
print(response)
```

## Supported Models
All models listed here https://jina.ai/embeddings/ are supported
