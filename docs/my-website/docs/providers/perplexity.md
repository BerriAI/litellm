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
    model="mistral-7b-instruct", 
    messages=messages,
    api_base="https://api.perplexity.ai"
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['PERPLEXITYAI_API_KEY'] = ""
response = completion(
    model="mistral-7b-instruct", 
    messages=messages,
    api_base="https://api.perplexity.ai",
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models
All models listed here https://docs.perplexity.ai/docs/model-cards are supported

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| codellama-34b-instruct   | `completion(model="codellama-34b-instruct", messages, api_base="https://api.perplexity.ai")`                                                         |
| llama-2-13b-chat         | `completion(model="llama-2-13b-chat", messages, api_base="https://api.perplexity.ai")`                                                               |
| llama-2-70b-chat         | `completion(model="llama-2-70b-chat", messages, api_base="https://api.perplexity.ai")`                                                               |
| mistral-7b-instruct      | `completion(model="mistral-7b-instruct", messages, api_base="https://api.perplexity.ai")`                                                            |
| replit-code-v1.5-3b     | `completion(model="replit-code-v1.5-3b", messages, api_base="https://api.perplexity.ai")`                                                            |





