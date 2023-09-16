# /Get Model Context Window & Cost per token [100+ LLMs]

For every LLM LiteLLM allows you to:
* Get model context window 
* Get cost per token 

## LiteLLM Package 
Usage
```python
import litellm
model_data = litellm.model_cost["gpt-4"]
```

## LiteLLM API api.litellm.ai
Usage
```python
import requests

url = "https://api.litellm.ai/get_max_tokens?model=claude-2"

response = requests.request("GET", url)

print(response.text)
```

```curl
curl --location 'https://api.litellm.ai/get_max_tokens?model=gpt-3.5-turbo'
```