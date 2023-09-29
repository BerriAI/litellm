# NLP Cloud

LiteLLM supports all LLMs on NLP Cloud.

## API Keys

```python 
import os 

os.environ["NLP_CLOUD_API_KEY"] = "your-api-key"
```

## Sample Usage

```python
import os
from litellm import completion 

# set env
os.environ["NLP_CLOUD_API_KEY"] = "your-api-key" 

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="dolphin", messages=messages)
print(response)
```

## streaming 
Just set `stream=True` when calling completion.

```python
import os
from litellm import completion 

# set env
os.environ["NLP_CLOUD_API_KEY"] = "your-api-key" 

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="dolphin", messages=messages, stream=True)
for chunk in response:
    print(chunk["choices"][0]["delta"]["content"])  # same as openai format
```

## non-dolphin models 

By default, LiteLLM will map `dolphin` and `chatdolphin` to nlp cloud. 

If you're trying to call any other model (e.g. GPT-J, Llama-2, etc.) with nlp cloud, just set it as your custom llm provider. 


```python
import os
from litellm import completion 

# set env - [OPTIONAL] replace with your nlp cloud key
os.environ["NLP_CLOUD_API_KEY"] = "your-api-key" 

messages = [{"role": "user", "content": "Hey! how's it going?"}]

# e.g. to call Llama2 on NLP Cloud
response = completion(model="nlp_cloud/finetuned-llama-2-70b", messages=messages, stream=True)
for chunk in response:
    print(chunk["choices"][0]["delta"]["content"])  # same as openai format
```
