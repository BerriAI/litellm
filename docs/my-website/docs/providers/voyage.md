# Voyage AI
https://docs.voyageai.com/embeddings/

## API Key
```python
# env variable
os.environ['VOYAGE_API_KEY']
```

## Sample Usage - Embedding
```python
from litellm import embedding
import os

os.environ['VOYAGE_API_KEY'] = ""
response = embedding(
    model="voyage/voyage-01",
    input=["good morning from litellm"],
)
print(response)
```

## Supported Models
All models listed here https://docs.voyageai.com/embeddings/#models-and-specifics are supported

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| voyage-2 | `embedding(model="voyage/voyage-2", input)` | 
| voyage-large-2 | `embedding(model="voyage/voyage-large-2", input)` | 
| voyage-law-2 | `embedding(model="voyage/voyage-law-2", input)` | 
| voyage-code-2 | `embedding(model="voyage/voyage-code-2", input)` | 
| voyage-lite-02-instruct | `embedding(model="voyage/voyage-lite-02-instruct", input)` | 
| voyage-01 | `embedding(model="voyage/voyage-01", input)` | 
| voyage-lite-01 | `embedding(model="voyage/voyage-lite-01", input)` | 
| voyage-lite-01-instruct | `embedding(model="voyage/voyage-lite-01-instruct", input)` | 