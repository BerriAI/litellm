# Voyage AI
https://docs.voyageai.com/embeddings/

## Proxy settings

| Model Settings | Value                                                        |
| -------------- |--------------------------------------------------------------|
| Provider | OpenAI-Compatible Endpoint                                   |
| API Base | https://api.voyageai.com/v1                                  |
| Model Name | e.g. voyage-3-large                                          |
| LiteLLM Model Name | e.g. voyage/voyage-3-large                                   |
| Custom LLM Provider | voyage (can not be set on creation, only on model edit page) | 


:: info

  It's important that the API Base includes the `/v1` postfix and that the LiteLLM Model Name includes the `voyage/` prefix.

::


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
    model="voyage/voyage-3-large",
    input=["good morning from litellm"],
)
print(response)
```

## Supported Models
All models listed here https://docs.voyageai.com/embeddings/#models-and-specifics are supported

| Model Name              | Function Call                                              |
|-------------------------|------------------------------------------------------------|
| voyage-3-large          | `embedding(model="voyage/voyage-3-large", input)`          | 
| voyage-3                | `embedding(model="voyage/voyage-3", input)`                | 
| voyage-3-lite           | `embedding(model="voyage/voyage-3-lite", input)`           | 
| voyage-code-3           | `embedding(model="voyage/voyage-code-3", input)`           | 
| voyage-finance-2        | `embedding(model="voyage/voyage-finance-2", input)`        | 
| voyage-law-2            | `embedding(model="voyage/voyage-law-2", input)`            | 
| voyage-code-2           | `embedding(model="voyage/voyage-code-2", input)`           | 
| voyage-multilingual-2   | `embedding(model="voyage/voyage-multilingual-2	", input)`  | 
| voyage-large-2-instruct | `embedding(model="voyage/voyage-large-2-instruct", input)` | 
| voyage-large-2          | `embedding(model="voyage/voyage-large-2", input)`          |
| voyage-2           | `embedding(model="voyage/voyage-2", input)`           | 
| voyage-lite-02-instruct | `embedding(model="voyage/voyage-lite-02-instruct", input)` | 
| voyage-01          | `embedding(model="voyage/voyage-01", input)`          | 
| voyage-lite-01     | `embedding(model="voyage/voyage-lite-01", input)`     | 
| voyage-lite-01-instruct | `embedding(model="voyage/voyage-lite-01-instruct", input)` | 
