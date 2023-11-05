# Setting API Keys, Base, Version

LiteLLM allows you to specify the following:
* API Key
* API Base
* API Version
* API Type

Useful Helper functions: 
* [`check_valid_key()`](#check_valid_key)
* [`get_valid_models()`](#get_valid_models)

You can set the API configs using:
* Environment Variables
* litellm variables `litellm.api_key`
* Passing args to `completion()`

## Environment Variables

### Setting API Keys

Set the liteLLM API key or specific provider key:

```python
import os 

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "Your API Key"
os.environ["ANTHROPIC_API_KEY"] = "Your API Key"
os.environ["REPLICATE_API_KEY"] = "Your API Key"
os.environ["TOGETHERAI_API_KEY"] = "Your API Key"
```

### Setting API Base, API Version, API Type

```python
# for azure openai
os.environ['AZURE_API_BASE'] = "https://openai-gpt-4-test2-v-12.openai.azure.com/"
os.environ['AZURE_API_VERSION'] = "2023-05-15" # [OPTIONAL]
os.environ['AZURE_API_TYPE'] = "azure" # [OPTIONAL]

# for openai
os.environ['OPENAI_API_BASE'] = "https://openai-gpt-4-test2-v-12.openai.azure.com/"
```

## litellm variables

### litellm.api_key
This variable is checked for all providers

```python
import litellm
# openai call
litellm.api_key = "sk-OpenAIKey"
response = litellm.completion(messages=messages, model="gpt-3.5-turbo")

# anthropic call
litellm.api_key = "sk-AnthropicKey"
response = litellm.completion(messages=messages, model="claude-2")
```

### litellm.provider_key (example litellm.openai_key)

```python
litellm.openai_key = "sk-OpenAIKey"
response = litellm.completion(messages=messages, model="gpt-3.5-turbo")

# anthropic call
litellm.anthropic_key = "sk-AnthropicKey"
response = litellm.completion(messages=messages, model="claude-2")
```

### litellm.api_base

```python
import litellm
litellm.api_base = "https://hosted-llm-api.co"
response = litellm.completion(messages=messages, model="gpt-3.5-turbo")
```

### litellm.api_version

```python
import litellm
litellm.api_version = "2023-05-15"
response = litellm.completion(messages=messages, model="gpt-3.5-turbo")
```

### litellm.organization
```python
import litellm
litellm.organization = "LiteLlmOrg"
response = litellm.completion(messages=messages, model="gpt-3.5-turbo")
```

## Passing Args to completion()

You can pass the API key within `completion()` call:

### api_key
```python
from litellm import completion

messages = [{ "content": "Hello, how are you?","role": "user"}]

response = completion("command-nightly", messages, api_key="Your-Api-Key")
```

### api_base

```python
from litellm import completion

messages = [{ "content": "Hello, how are you?","role": "user"}]

response = completion("command-nightly", messages, api_base="https://hosted-llm-api.co")
```

### api_version

```python
from litellm import completion

messages = [{ "content": "Hello, how are you?","role": "user"}]

response = completion("command-nightly", messages, api_version="2023-02-15")
```

## Helper Functions

### `check_valid_key()`

Check if a user submitted a valid key for the model they're trying to call. 

```python
key = "bad-key"
response = check_valid_key(model="gpt-3.5-turbo", api_key=key)
assert(response == False)
```

### `get_valid_models()`

This helper reads the .env and returns a list of supported llms for user

```python
old_environ = os.environ
os.environ = {'OPENAI_API_KEY': 'temp'} # mock set only openai key in environ

valid_models = get_valid_models()
print(valid_models)

# list of openai supported llms on litellm
expected_models = litellm.open_ai_chat_completion_models + litellm.open_ai_text_completion_models

assert(valid_models == expected_models)

# reset replicate env key
os.environ = old_environ
```

### `validate_environment(model: str)`

This helper tells you if you have all the required environment variables for a model, and if not - what's missing. 

```python
from litellm import validate_environment

print(validate_environment("openai/gpt-3.5-turbo"))
```