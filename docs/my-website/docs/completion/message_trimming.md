# Trimming Messages - litellm.safe_messages
**Use litellm.safe_messages() to ensure messages does not exceed a model's token limit or specified `max_tokens`**

## Usage 
```python
from litellm import completion
from litellm.utils import safe_messages

response = completion(
  model=model,
  messages=safe_messages(messages, model) # safe_messages ensures tokens(messages) < tokens (model)
)

```

## Parameters

The function uses the following parameters:

- `messages`: This should be a list of input messages 

- `model`: This is the LiteLLM model being used. This parameter is optional, as you can alternatively specify the `max_tokens` parameter.

- `system_message`: This is a string containing an optional system message that will be preserved at the beginning of the conversation. This parameter is optional and set to `None` by default.

- `trim_ratio`: This represents the target ratio of tokens to use following trimming. It's default value is 0.75, which implies that messages will be trimmed to utilise about 75%