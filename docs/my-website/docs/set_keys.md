# Setting API Keys, Base, Version

LiteLLM allows you to specify the following:
* API Key
* API Base
* API Version
* API Type

You can set the API configs using:
* Environment Variables
* litellm variables `litellm.api_key`
* Passing args to `completion()`

# Setting API Keys, Base, and Version 

API keys, base API endpoint, and API version can be set via environment variables or passed dynamically. 

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
os.environ['AZURE_API_VERSION'] = "2023-05-15"
os.environ['AZURE_API_TYPE'] = "your-custom-type"

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
se = litellm.completion(messages=messages, model="gpt-3.5-turbo")
```



You can pass the API key within `completion()` call:

```python
from litellm import completion

messages = [{ "content": "Hello, how are you?","role": "user"}]

response = completion("gpt-3.5-turbo", messages, api_key="Your-Api-Key")
```

Sample usage of liteLLM with OpenAI:

```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "Your API Key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion("gpt-3.5-turbo", messages)
```

