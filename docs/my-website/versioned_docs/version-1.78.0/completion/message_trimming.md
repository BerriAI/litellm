# Trimming Input Messages
**Use litellm.trim_messages() to ensure messages does not exceed a model's token limit or specified `max_tokens`**

## Usage 
```python
from litellm import completion
from litellm.utils import trim_messages

response = completion(
    model=model, 
    messages=trim_messages(messages, model) # trim_messages ensures tokens(messages) < max_tokens(model)
) 
```

## Usage - set max_tokens
```python
from litellm import completion
from litellm.utils import trim_messages

response = completion(
    model=model, 
    messages=trim_messages(messages, model, max_tokens=10), # trim_messages ensures tokens(messages) < max_tokens
) 
```

## Parameters

The function uses the following parameters:

- `messages`:[Required] This should be a list of input messages 

- `model`:[Optional] This is the LiteLLM model being used. This parameter is optional, as you can alternatively specify the `max_tokens` parameter.

- `max_tokens`:[Optional] This is an int, manually set upper limit on messages

- `trim_ratio`:[Optional] This represents the target ratio of tokens to use following trimming. It's default value is 0.75, which implies that messages will be trimmed to utilise about 75%