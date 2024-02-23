# Groq
https://groq.com/

## API Key
```python
# env variable
os.environ['GROQ_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['GROQ_API_KEY'] = ""
response = completion(
    model="groq/llama2-70b-4096", 
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

os.environ['GROQ_API_KEY'] = ""
response = completion(
    model="groq/llama2-70b-4096", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models - ALL Groq Models Supported!
We support ALL Groq models, just set `groq/` as a prefix when sending completion requests

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| llama2-70b-4096 | `completion(model="groq/llama2-70b-4096", messages)` | 
