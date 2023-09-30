# OpenAI
LiteLLM supports OpenAI Chat + Text completion and embedding calls.

### API Keys

```python
import os 

os.environ["OPENAI_API_KEY"] = "your-api-key"
```
**Need a dedicated key?**
Email us @ krrish@berri.ai 

[**See all supported models by the litellm api key**](../proxy_api.md#supported-models-for-litellm-key)

### Usage
```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"


messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion("gpt-3.5-turbo", messages)
```

### OpenAI Chat Completion Models

| Model Name       | Function Call                          | Required OS Variables                |
|------------------|----------------------------------------|--------------------------------------|
| gpt-3.5-turbo    | `completion('gpt-3.5-turbo', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-0301    | `completion('gpt-3.5-turbo-0301', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-0613    | `completion('gpt-3.5-turbo-0613', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-16k    | `completion('gpt-3.5-turbo-16k', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-16k-0613    | `completion('gpt-3.5-turbo-16k-0613', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-4            | `completion('gpt-4', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-0314            | `completion('gpt-4-0314', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-0613            | `completion('gpt-4-0613', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-32k            | `completion('gpt-4-32k', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-32k-0314            | `completion('gpt-4-32k-0314', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-32k-0613            | `completion('gpt-4-32k-0613', messages)`         | `os.environ['OPENAI_API_KEY']`       |

These also support the `OPENAI_API_BASE` environment variable, which can be used to specify a custom API endpoint.

### OpenAI Text Completion Models / Instruct Models

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| gpt-3.5-turbo-instruct | `completion('gpt-3.5-turbo-instruct', messages)` | `os.environ['OPENAI_API_KEY'`       |
| text-davinci-003 | `completion('text-davinci-003', messages)` | `os.environ['OPENAI_API_KEY']`       |
| ada-001 | `completion('ada-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| curie-001 | `completion('curie-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| babbage-001 | `completion('babbage-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| babbage-002 | `completion('ada-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| davinci-002 | `completion('davinci-002', messages)` | `os.environ['OPENAI_API_KEY']`       |


### Using Helicone Proxy with LiteLLM
```python
import os 
import litellm
from litellm import completion

os.environ["OPENAI_API_KEY"] = ""

# os.environ["OPENAI_API_BASE"] = ""
litellm.api_base = "https://oai.hconeai.com/v1"
litellm.headers = {
    "Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}",
    "Helicone-Cache-Enabled": "true",
}

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion("gpt-3.5-turbo", messages)
```

### Using OpenAI Proxy with LiteLLM
```python
import os 
import litellm
from litellm import completion

os.environ["OPENAI_API_KEY"] = ""

# set custom api base to your proxy
# either set .env or litellm.api_base
# os.environ["OPENAI_API_BASE"] = ""
litellm.api_base = "your-openai-proxy-url"


messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion("openai/your-model-name", messages)
```

If you need to set api_base dynamically, just pass it in completions instead - `completions(...,api_base="your-proxy-api-base")`

For more check out [setting API Base/Keys](../set_keys.md)