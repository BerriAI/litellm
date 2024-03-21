# ZhipuAI
## API Keys
api_key can be passed directly to `litellm.completion` or set as `litellm.api_key` params
```python
import os
os.environ["ZHIPU_API_KEY"] = "" # "my-zhipu-api-key"
```

## Usage

### Completion - using .env variables

```python
from litellm import completion

## set ENV variables
os.environ["ZHIPU_API_KEY"] = ""

# zhipuai call
response = completion(
    model = "zhipuai/glm-3-turbo",
    messages = [{"role": "user", "content": "Hello, how are you?"}]
)
```

### Completion - using api_key

```python
import litellm

# zhipuai call
response = litellm.completion(
    model = "zhipuai/glm-3-turbo",
    api_key = "",
    messages = [{"role": "user", "content": "good morning"}],
)
```

## ZhipuAI Chat Completion Models

| Model Name       | Function Call                          |
|------------------|----------------------------------------|
| glm-4            | `completion('zhipuai/glm-4', messages)`         |
| glm-4v            | `completion('zhipuai/glm-4v', messages)`         |
| glm-3-turbo            | `completion('zhipuai/glm-3-turbo', messages)`         |

## Image Generation
| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
| cogview-3   | `response = image_generation(model="zhipuai/cogview-3", prompt=prompt)` |

#### Usage
```python
import os 
from litellm import image_generation

os.environ["ZHIPU_API_KEY"] = ""

# zhipuai call
response = image_generation(
    model = "zhipuai/cogview-3", 
    prompt = "A cute baby sea otter",
)
```

## Embedding Models
| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
| embedding-2   | `response = embedding(model="zhipuai/embedding-2", input=input)` |

#### Usage
```python
import os 
from litellm import embedding

os.environ["ZHIPU_API_KEY"] = ""

# zhipuai call
response = embedding(
    model = "zhipuai/embedding-2", 
    input = "good morning from litellm",
)

```
