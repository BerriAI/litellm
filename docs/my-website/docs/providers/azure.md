# Azure
LiteLLM supports Azure Chat + Embedding calls. 

### API KEYS
```python
import os

os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""
```


### Azure OpenAI Chat Completion Models

```python

from litellm import completion

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]

# azure call
response = completion("azure/<your_deployment_name>", messages)
```

### Set API Key, API Base, API Version in Completion()

```python
import litellm
response = litellm.completion(
    model = "azure/<your deployment name>",             # model = azure/<your deployment name> 
    api_base = "",                                      # azure api base
    api_version = "",                                   # azure api version
    api_key = "",                                       # azure api key
    messages = [{"role": "user", "content": "good morning"}],
)
```