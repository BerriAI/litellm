# Setting API Keys, Base, Version

LiteLLM allows you to specify the following:
* API Key
* API Base
* API Version
* API Type

You can set the API configs using:
* Environment Variables
* litellm.api_key [litellm variables]
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

### Setting API Base and API Version

```python
AZURE_API_BASE = "https://openai-gpt-4-test-v-1.openai.azure.com/"
AZURE_API_VERSION = "2023-05-15"
```

### Setting API Version

## Dynamic API Key

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

