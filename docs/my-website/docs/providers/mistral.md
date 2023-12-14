# Mistral AI API
https://docs.mistral.ai/api/

## API Key
```python
# env variable
os.environ['MISTRAL_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['MISTRAL_API_KEY'] = ""
response = completion(
    model="mistral/mistral-tiny"", 
    messages=[
        "role": "user",
        "content": "hello from litellm"
    ]
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['MISTRAL_API_KEY'] = ""
response = completion(
    model="mistral/mistral-tiny", 
    messages=[
        "role": "user",
        "content": "hello from litellm"
    ]
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models
All models listed here https://docs.mistral.ai/platform/endpoints are supported. We actively maintain the list of models, pricing, token window, etc. [here](https://github.com/BerriAI/litellm/blob/c1b25538277206b9f00de5254d80d6a83bb19a29/model_prices_and_context_window.json).

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mistral-tiny | `completion(model="mistral/mistral-tiny", messages)` | 
| mistral-small | `completion(model="mistral/mistral-small", messages)` | 
| mistral-medium | `completion(model="mistral/mistral-medium", messages)` | 





