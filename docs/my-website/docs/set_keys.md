# Setting API Keys, Base, Version


environment variables

setting litellm.api_key

setting litellm.<provider_name>_api_key (e.g. litellm.openai_api_key, litellm.anthropic_api_key, etc.)

passing in dynamically via completion() call - e.g. completion(..., api_key="...")


# Setting API Keys, Base, and Version 

API keys, base API endpoint, and API version can be set via environment variables or passed dynamically. 

## Environment Variables

Set the liteLLM API key or specific provider key:

```python
import os 

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "Your API Key"
os.environ["ANTHROPIC_API_KEY"] = "Your API Key"
os.environ["REPLICATE_API_KEY"] = "Your API Key"
os.environ["TOGETHERAI_API_KEY"] = "Your API Key"
```

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

