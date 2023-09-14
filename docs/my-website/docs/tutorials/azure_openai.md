# Call Azure OpenAI, OpenAI Using LiteLLM

### Usage
```python
import os 
from litellm import completion

# openai configs
os.environ["OPENAI_API_KEY"] = ""

# azure openai configs
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = "https://openai-gpt-4-test-v-1.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2023-05-15"



# openai call
response = completion(
    model = "gpt-3.5-turbo", 
    messages= = [{ "content": "Hello, how are you?","role": "user"}]
)

# azure call
response = completion(
    model = "azure/<your-azure-deployment>",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

```