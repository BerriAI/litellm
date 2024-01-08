# Xinference [Xorbits Inference]
https://inference.readthedocs.io/en/latest/index.html

## API Base, Key
```python
# env variable
os.environ['XINFERENCE_API_BASE'] = "http://127.0.0.1:9997/v1"
os.environ['XINFERENCE_API_KEY'] = "anything" #[optional] no api key required
```

## Sample Usage - Embedding
```python
from litellm import embedding
import os

os.environ['XINFERENCE_API_BASE'] = "http://127.0.0.1:9997/v1"
response = embedding(
    model="xinference/bge-base-en",
    input=["good morning from litellm"],
)
print(response)
```

## Sample Usage `api_base` param
```python
from litellm import embedding
import os

response = embedding(
    model="xinference/bge-base-en",
    api_base="http://127.0.0.1:9997/v1",
    input=["good morning from litellm"],
)
print(response)
```

## Supported Models
All models listed here https://inference.readthedocs.io/en/latest/models/builtin/embedding/index.html are supported

| Model Name                   | Function Call                                          |
|------------------------------|--------------------------------------------------------|
| bge-base-en                  | `embedding(model="xinference/bge-base-en", input)`                  |
| bge-base-en-v1.5             | `embedding(model="xinference/bge-base-en-v1.5", input)`             |
| bge-base-zh                  | `embedding(model="xinference/bge-base-zh", input)`                  |
| bge-base-zh-v1.5             | `embedding(model="xinference/bge-base-zh-v1.5", input)`             |
| bge-large-en                 | `embedding(model="xinference/bge-large-en", input)`                 |
| bge-large-en-v1.5            | `embedding(model="xinference/bge-large-en-v1.5", input)`            |
| bge-large-zh                 | `embedding(model="xinference/bge-large-zh", input)`                 |
| bge-large-zh-noinstruct      | `embedding(model="xinference/bge-large-zh-noinstruct", input)`      |
| bge-large-zh-v1.5            | `embedding(model="xinference/bge-large-zh-v1.5", input)`            |
| bge-small-en-v1.5            | `embedding(model="xinference/bge-small-en-v1.5", input)`            |
| bge-small-zh                 | `embedding(model="xinference/bge-small-zh", input)`                 |
| bge-small-zh-v1.5            | `embedding(model="xinference/bge-small-zh-v1.5", input)`            |
| e5-large-v2                  | `embedding(model="xinference/e5-large-v2", input)`                  |
| gte-base                     | `embedding(model="xinference/gte-base", input)`                     |
| gte-large                    | `embedding(model="xinference/gte-large", input)`                    |
| jina-embeddings-v2-base-en   | `embedding(model="xinference/jina-embeddings-v2-base-en", input)`   |
| jina-embeddings-v2-small-en  | `embedding(model="xinference/jina-embeddings-v2-small-en", input)`  |
| multilingual-e5-large        | `embedding(model="xinference/multilingual-e5-large", input)`        |



