# SOLAR 

LiteLLM supports from [Solar](https://developers.upstage.ai/product-guides/solar).

### API KEYS
You can set the API keys in the following ways:

```python
import os 
os.environ["SOLAR_API_KEY"] = ""
```


```python
import litellm 
litllm.solar_api_key = ""
```


### Usage

```python
import litellm 
import os

# set the secrets from https://console.upstage.ai/services/solar
os.environ["SOLAR_API_KEY"] = "DUMMY"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# solar call
response = completion(model="solar-1-mini-chat", messages=messages)
response
```