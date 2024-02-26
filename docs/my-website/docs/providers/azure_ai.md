# Azure AI Studio

## Using Mistral models deployed on Azure AI Studio

**Ensure you have the `/v1` in your api_base**

### Sample Usage
```python
from litellm import completion
import os

response = completion(
    model="mistral/Mistral-large-dfgfj", 
    api_base="https://Mistral-large-dfgfj-serverless.eastus2.inference.ai.azure.com/v1",
    api_key = "JGbKodRcTp****"
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

### [LiteLLM Proxy] Using Mistral Models 

Set this on your litellm proxy config.yaml

**Ensure you have the `/v1` in your api_base**
```yaml
model_list:
  - model_name: mistral
    litellm_params:
      model: mistral/Mistral-large-dfgfj
      api_base: https://Mistral-large-dfgfj-serverless.eastus2.inference.ai.azure.com/v1
      api_key: JGbKodRcTp****
```


