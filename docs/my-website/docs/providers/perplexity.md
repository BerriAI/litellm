# Perplexity AI (pplx-api)
https://www.perplexity.ai

## API Key
```python
# env variable
os.environ['PERPLEXITYAI_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/mistral-7b-instruct", 
    messages=messages
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="perplexity/mistral-7b-instruct", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models
All models listed here https://docs.perplexity.ai/docs/model-cards are supported

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| codellama-34b-instruct | `completion(model="perplexity/codellama-34b-instruct", messages)` | 
| llama-2-13b-chat | `completion(model="perplexity/llama-2-13b-chat", messages)` | 
| llama-2-70b-chat | `completion(model="perplexity/llama-2-70b-chat", messages)` | 
| mistral-7b-instruct | `completion(model="perplexity/mistral-7b-instruct", messages)` | 
| replit-code-v1.5-3b | `completion(model="perplexity/replit-code-v1.5-3b", messages)` |





