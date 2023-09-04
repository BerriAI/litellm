# Azure
LiteLLM supports Azure Chat + Embedding calls. 

### API KEYS
```
import os 

os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""
```


### Azure OpenAI Chat Completion Models

```python

from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["AZURE_API_KEY"] = "azure key"


messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# azure call
response = completion("azure/<your_deployment_id>", messages)
```