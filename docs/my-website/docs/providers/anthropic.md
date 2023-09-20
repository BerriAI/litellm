# Anthropic
LiteLLM supports Claude-1, 1.2 and Claude-2.

## API Keys
We provide a free $10 community-key for testing all providers on LiteLLM. You can replace this with your own key. 

```python 
import os 

os.environ["ANTHROPIC_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your anthropic key
```

## Sample Usage

```python
import os
from litellm import completion 

# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" 

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude-instant-1", messages=messages)
print(response)
```

## streaming 
Just set `stream=True` when calling completion.

```python
import os
from litellm import completion 

# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" 

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude-instant-1", messages=messages, stream=True)
for chunk in response:
    print(chunk["choices"][0]["delta"]["content"])  # same as openai format
```


### Model Details

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| claude-instant-1  | `completion('claude-instant-1', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-instant-1.2  | `completion('claude-instant-1.2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-2  | `completion('claude-2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
