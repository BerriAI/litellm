# Fireworks AI
https://fireworks.ai/

**We support ALL Fireworks AI models, just set `fireworks_ai/` as a prefix when sending completion requests**

## API Key
```python
# env variable
os.environ['FIREWORKS_AI_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['FIREWORKS_AI_API_KEY'] = ""
response = completion(
    model="fireworks_ai/mixtral-8x7b-instruct", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['FIREWORKS_AI_API_KEY'] = ""
response = completion(
    model="fireworks_ai/mixtral-8x7b-instruct", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models - ALL Fireworks AI Models Supported!
We support ALL Fireworks AI models, just set `fireworks_ai/` as a prefix when sending completion requests

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mixtral-8x7b-instruct | `completion(model="fireworks_ai/mixtral-8x7b-instruct", messages)` | 
| firefunction-v1 | `completion(model="fireworks_ai/firefunction-v1", messages)` |
| llama-v2-70b-chat | `completion(model="fireworks_ai/llama-v2-70b-chat", messages)` |  