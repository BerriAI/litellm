# OpenAI
LiteLLM supports OpenAI Chat + Text completion and embedding calls.

### API Keys

```python
import os 
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_ORGANIZATION"] = "your-org-id" # OPTIONAL
```

### Usage
```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"

# openai call
response = completion(
    model = "gpt-3.5-turbo", 
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

### OpenAI Chat Completion Models

| Model Name            | Function Call                                                   |
|-----------------------|-----------------------------------------------------------------|
| gpt-3.5-turbo         | `response = completion(model="gpt-3.5-turbo", messages=messages)` |
| gpt-3.5-turbo-0301    | `response = completion(model="gpt-3.5-turbo-0301", messages=messages)` |
| gpt-3.5-turbo-0613    | `response = completion(model="gpt-3.5-turbo-0613", messages=messages)` |
| gpt-3.5-turbo-16k     | `response = completion(model="gpt-3.5-turbo-16k", messages=messages)` |
| gpt-3.5-turbo-16k-0613| `response = completion(model="gpt-3.5-turbo-16k-0613", messages=messages)` |
| gpt-4                 | `response = completion(model="gpt-4", messages=messages)` |
| gpt-4-0314            | `response = completion(model="gpt-4-0314", messages=messages)` |
| gpt-4-0613            | `response = completion(model="gpt-4-0613", messages=messages)` |
| gpt-4-32k             | `response = completion(model="gpt-4-32k", messages=messages)` |
| gpt-4-32k-0314        | `response = completion(model="gpt-4-32k-0314", messages=messages)` |
| gpt-4-32k-0613        | `response = completion(model="gpt-4-32k-0613", messages=messages)` |


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